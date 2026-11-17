# 🛡️ Job Sentinel — Agente Inteligente de Monitoramento de Processos Seletivos

Pipeline automatizado em Python que monitora uma caixa de entrada via **IMAP**, filtra ruído comercial localmente e utiliza **LLMs (Google Gemini)** com **Structured Outputs (Pydantic)** para identificar atualizações reais de recrutamento (entrevistas, testes técnicos, feedbacks) e enviar alertas instantâneos via **Telegram**.

---

## 🏗️ Arquitetura e Fluxo de Dados

```text
[Gmail IMAP (BODY.PEEK[])] 
       │
       ▼
[Deduplicação via Hash SHA-256] ──► Já analisado? ──► Ignora
       │ (Novo)
       ▼
[Filtro Heurístico Local (PT/EN/ATS)] ──► Ruído comercial? ──► Descarta (0 tokens)
       │ (Candidato a vaga)
       ▼
[Gemini API + Pydantic Schema] ──► Fallback automático (3.8-flash → 3.5-flash → 2.5-flash-lite)
       │ (vaga_relevante == True)
       ▼
[Telegram Bot API] ──► Notificação formatada em tempo real