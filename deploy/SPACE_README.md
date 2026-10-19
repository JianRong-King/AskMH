---
title: AskMH Chatbot
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: RAG FAQ assistant answering strictly from the Malaysia Airlines AskMH knowledge base
---

# AskMH Chatbot

A retrieval-augmented FAQ assistant for Malaysia Airlines passengers. It answers only from the
public [AskMH knowledge base](https://www.askmh.malaysiaairlines.com/faq/s/), cites the article
it used, asks a clarifying question when the answer depends on details you have not given, and
hands off to (simulated) live chat when it cannot help.

Sign-in is required because the demo runs on a shared free-tier model quota. Your Google
account identifies the session and nothing else; no profile data is stored.

Source and engineering write-up: https://github.com/JianRong-King/askmh-chatbot

Non-commercial portfolio project. The FAQ content belongs to Malaysia Airlines.
