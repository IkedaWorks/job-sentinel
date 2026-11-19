import os
import time
import hashlib
import imaplib
import email
from email.header import decode_header
from email.message import Message
from datetime import datetime, timedelta
from typing import Optional
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# ==============================================================================
# 1. CONFIGURAÇÃO E VARIÁVEIS DE AMBIENTE
# ==============================================================================
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_APP_PASSWORD") or os.getenv("EMAIL_PASS", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_MODELS",
        "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b",
    ).split(",")
    if m.strip()
]
GEMINI_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.5-flash-lite,gemini-3.5-flash",
    ).split(",")
    if m.strip()
]

REQUEST_DELAY_SECONDS = 3
MAX_BODY_CHARS = 800
MAX_EMAILS_PER_RUN = 15
HISTORY_FILE = "processed.txt"

# ==============================================================================
# 2. FILTROS HEURÍSTICOS LOCAIS (CUSTO ZERO DE IA)
# ==============================================================================
NOISE_SENDERS = [
    "noreply@github.com",
    "notifications@github.com",
]

NOISE_SUBJECT_TERMS = [
    "[ikedaworks/job-sentinel]",
    "run failed:",
    "workflow run",
]

BLOCKED_KEYWORDS = [
    "pedido confirmado", "seu pedido", "saiu para entrega", "nota fiscal",
    "fatura", "pix", "boleto", "extrato", "cupom", "combinho", "black friday"
]

CAREER_KEYWORDS = [
    # Português
    "vaga", "entrevista", "processo seletivo", "candidatura", "recrutamento",
    "rh", "recursos humanos", "estágio", "estagio", "trainee", "oportunidade",
    "carreira", "desenvolvedor", "engenheiro", "feedback", "teste técnico",
    "proposta", "contratação", "seleção", "currículo", "curriculo",

    # Inglês
    "job", "role", "position", "opening", "opportunity", "career", "hiring",
    "recruiter", "recruiting", "recruitment", "talent", "headhunter",
    "interview", "screening", "chat", "call", "schedule", "availability",
    "application", "applied", "candidate", "candidacy", "resume", "cv",
    "assessment", "challenge", "coding", "take-home", "technical",
    "offer", "compensation", "onboarding", "next steps", "update",
    "background", "profile", "reach out", "reaching out", "connect",
    "engineer", "developer", "software", "intern", "internship", "remote",

    # Plataformas ATS
    "gupy", "linkedin", "greenhouse", "lever", "workday", "ashby",
    "breezy", "smartrecruiters", "icims", "taleo", "jobvite", "bamboohr",
    "deel", "remote.com", "turing", "toptal", "bairesdev", "andela",
    "revelo", "geekhunter", "programathor", "kenoby", "solides", "catho",
    "vagas.com", "wellfound", "angel.co", "glassdoor", "indeed"
]


# ==============================================================================
# 3. CONTRATO DE SAÍDA ESTRUTURADA (PYDANTIC)
# ==============================================================================
class JobAnalysis(BaseModel):
    vaga_relevante: bool
    empresa: str
    cargo: str
    status: str
    resumo_acao: str


# ==============================================================================
# 4. VALIDAÇÃO E CONTROLE DE IDEMPOTÊNCIA (HASH SHA-256)
# ==============================================================================
def validate_environment() -> None:
    """Garante que todas as credenciais obrigatórias estão carregadas (Fail-Fast)."""
    required_vars = {
        "EMAIL_USER": EMAIL_USER,
        "EMAIL_APP_PASSWORD": EMAIL_PASS,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }
    missing = [key for key, value in required_vars.items() if not value]
    if missing:
        raise EnvironmentError(f"Variáveis de ambiente ausentes: {', '.join(missing)}")

    if not GROQ_API_KEY and not GEMINI_API_KEY:
        raise EnvironmentError(
            "Configure ao menos uma chave de IA: GROQ_API_KEY ou GEMINI_API_KEY."
        )


def load_processed_hashes() -> set[str]:
    """Lê os hashes SHA-256 dos e-mails já processados anteriormente."""
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def save_processed_hash(email_hash: str) -> None:
    """Registra apenas o hash anônimo (sem expor assunto/remetente no repositório)."""
    with open(HISTORY_FILE, "a", encoding="utf-8") as file:
        file.write(f"{email_hash}\n")


def compute_email_hash(message: Message, sender: str, subject: str) -> str:
    """Gera assinatura única SHA-256 baseada no Message-ID oficial do e-mail."""
    message_id = message.get("Message-ID") or f"{sender}-{subject}-{message.get('Date', '')}"
    return hashlib.sha256(message_id.encode("utf-8", errors="ignore")).hexdigest()


# ==============================================================================
# 5. PARSING E LIMPEZA DE E-MAILS
# ==============================================================================
def decode_mime_header(raw_header: str | None) -> str:
    """Converte cabeçalhos MIME codificados para texto legível."""
    if not raw_header:
        return ""
    decoded_parts = decode_header(raw_header)
    fragments: list[str] = []
    for content, encoding in decoded_parts:
        if isinstance(content, bytes):
            fragments.append(content.decode(encoding or "utf-8", errors="ignore"))
        else:
            fragments.append(content)
    return "".join(fragments)


def extract_compact_body(message: Message) -> str:
    """Extrai o texto puro (text/plain) e limita o tamanho para economia de tokens."""
    raw_text = ""
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    raw_text = payload.decode("utf-8", errors="ignore")
                    break
    else:
        payload = message.get_payload(decode=True)
        if payload:
            raw_text = payload.decode("utf-8", errors="ignore")

    clean_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    compact_text = " ".join(clean_lines)
    return compact_text[:MAX_BODY_CHARS]


def is_job_candidate(sender: str, subject: str, body_snippet: str) -> bool:
    """Filtro heurístico local (0 tokens) bloqueando ruído e priorizando carreira."""
    sender_lower = sender.lower()
    subject_lower = subject.lower()

    if any(noise in sender_lower for noise in NOISE_SENDERS):
        return False

    if any(term in subject_lower for term in NOISE_SUBJECT_TERMS):
        return False

    searchable_text = f"{sender_lower} {subject_lower} {body_snippet[:300].lower()}"

    if any(keyword in searchable_text for keyword in CAREER_KEYWORDS):
        return True

    if any(keyword in searchable_text for keyword in BLOCKED_KEYWORDS):
        return False

    return False


# ==============================================================================
# 6. ANÁLISE SEMÂNTICA COM GROQ + GEMINI + FALLBACK DE RESILIÊNCIA
# ==============================================================================
def build_classification_prompt(sender: str, subject: str, body_snippet: str) -> str:
    """Monta o prompt padronizado exigindo JSON estrito compatível com JobAnalysis."""
    return f"""
    Classifique se este e-mail é uma atualização real de processo seletivo ou oportunidade de emprego
    (confirmação de candidatura, entrevista, teste técnico, feedback ou contato de recrutador).
    Ignore cursos pagos, propaganda, notificações de repositório ou alertas genéricos.

    Responda EXCLUSIVAMENTE com um objeto JSON válido contendo exatamente estas chaves:
    {{
      "vaga_relevante": true ou false,
      "empresa": "nome da empresa",
      "cargo": "cargo ou vaga",
      "status": "etapa do processo seletivo",
      "resumo_acao": "resumo curto em português"
    }}

    De: {sender}
    Assunto: {subject}
    Trecho: {body_snippet}
    """.strip()


def try_classify_groq(prompt: str) -> Optional[JobAnalysis]:
    """Tenta classificar usando a API OpenAI-compatible da Groq em cascata."""
    if not GROQ_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for model_name in GROQ_MODELS:
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "Você é um assistente especialista em recrutamento que responde apenas em JSON válido.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            return JobAnalysis.model_validate_json(raw_content)
        except Exception as exc:
            print(f"   ⚠️ Groq ({model_name}) falhou ({type(exc).__name__}: {exc}). Tentando próximo...")
            continue

    return None


def try_classify_gemini(prompt: str) -> Optional[JobAnalysis]:
    """Tenta classificar usando a SDK do Google Gemini como provedor secundário."""
    if not GEMINI_API_KEY:
        return None

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=20_000),
    )

    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": JobAnalysis,
                },
            )
            return JobAnalysis.model_validate_json(response.text)
        except Exception as exc:
            print(f"   ⚠️ Gemini ({model_name}) falhou ({type(exc).__name__}: {exc}). Tentando próximo...")
            continue

    return None


def classify_email(sender: str, subject: str, body_snippet: str) -> JobAnalysis:
    """Orquestra a cadeia Groq -> Gemini -> Fallback preventivo seguro."""
    prompt = build_classification_prompt(sender, subject, body_snippet)

    groq_result = try_classify_groq(prompt)
    if groq_result is not None:
        return groq_result

    gemini_result = try_classify_gemini(prompt)
    if gemini_result is not None:
        return gemini_result

    print("   ⚠️ Todas as IAs indisponíveis no momento. Enviando alerta preventivo...")
    return JobAnalysis(
        vaga_relevante=True,
        empresa=sender[:40],
        cargo="Verificar no e-mail",
        status="Possível vaga (IA indisponível no momento)",
        resumo_acao=f"Assunto: {subject}",
    )


# ==============================================================================
# 7. NOTIFICAÇÃO VIA TELEGRAM BOT API
# ==============================================================================
def sanitize_markdown(text: str) -> str:
    """Remove caracteres que quebram a formatação Markdown do Telegram."""
    return str(text).replace("_", " ").replace("*", "").replace("`", "")


def send_telegram_alert(message_text: str) -> None:
    """Envia a notificação formatada via HTTP POST para o bot do Telegram."""
    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text,
        "parse_mode": "Markdown",
    }
    response = requests.post(api_url, json=payload, timeout=10)
    response.raise_for_status()


# ==============================================================================
# 8. ORQUESTRADOR PRINCIPAL
# ==============================================================================
def run_sentinel() -> None:
    validate_environment()
    print("Conectando ao Gmail via IMAP...")
    processed_hashes = load_processed_hashes()

    imap_conn = imaplib.IMAP4_SSL("imap.gmail.com")
    imap_conn.login(EMAIL_USER, EMAIL_PASS)
    imap_conn.select("inbox")

    since_date = (datetime.now() - timedelta(days=2)).strftime("%d-%b-%Y")
    _, search_data = imap_conn.search(None, f'(UNSEEN SINCE "{since_date}")')

    email_ids = search_data[0].split()
    if not email_ids:
        print("Nenhum e-mail novo correspondente ao filtro encontrado.")
        imap_conn.logout()
        return

    selected_ids = email_ids[-MAX_EMAILS_PER_RUN:]
    print(f"Encontrados {len(email_ids)} não lido(s). Analisando até {len(selected_ids)}...")

    ai_calls = 0
    for email_uid in selected_ids:
        _, raw_fetch = imap_conn.fetch(email_uid, "(BODY.PEEK[])")
        message = email.message_from_bytes(raw_fetch[0][1])

        sender = decode_mime_header(message.get("From"))
        subject = decode_mime_header(message.get("Subject"))
        email_hash = compute_email_hash(message, sender, subject)

        if email_hash in processed_hashes:
            continue

        body_snippet = extract_compact_body(message)

        if not is_job_candidate(sender, subject, body_snippet):
            print(f"🚫 Ignorado localmente (0 tokens gastos): {subject}")
            save_processed_hash(email_hash)
            processed_hashes.add(email_hash)
            continue

        if ai_calls > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

        print(f"-> Analisando trecho ({len(body_snippet)} chars) com IA: {subject}")
        analysis = classify_email(sender, subject, body_snippet)
        ai_calls += 1

        if analysis.vaga_relevante:
            alert_text = (
                f"🚨 *Atualização de Vaga!*\n\n"
                f"🏢 *Empresa:* {sanitize_markdown(analysis.empresa)}\n"
                f"💼 *Cargo:* {sanitize_markdown(analysis.cargo)}\n"
                f"📌 *Status:* {sanitize_markdown(analysis.status)}\n"
                f"📝 *Resumo:* {sanitize_markdown(analysis.resumo_acao)}"
            )
            send_telegram_alert(alert_text)
            print(f"✅ Alerta enviado para o Telegram ({analysis.empresa})!")
        else:
            print("⏭️ Ignorado pela IA (não é vaga/processo seletivo).")

        save_processed_hash(email_hash)
        processed_hashes.add(email_hash)

    imap_conn.logout()
    print(f"Verificação concluída. Total de chamadas à IA: {ai_calls}.")


if __name__ == "__main__":
    run_sentinel()