"""Kontrak Aman - Employment Contract Risk Checker for Fresh Graduates (Streamlit).

Pipeline: file ingestion -> structured clause extraction -> parser
-> risk-analysis LLM -> report.
"""
import io
import json
import os
import threading
from datetime import date
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gemini-3.5-flash-lite")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gemini-3.6-flash")
MAX_CHARS = 40_000
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "50"))
NOT_FOUND = "Tidak disebutkan dalam kontrak"
SAMPLE_PATH = Path(__file__).parent / "sample_kontrak.txt"

EXTRACT_PROMPT = """Anda adalah seorang konsultan ketenagakerjaan yang membantu fresh graduate memahami kontrak kerja mereka.

Dari teks kontrak kerja di bawah ini, identifikasi dan ekstrak informasi berikut jika ada:
1. Jenis Kontrak (PKWT / PKWTT / tidak disebutkan jelas)
2. Masa Percobaan (probation)
3. Gaji & Tunjangan (nominal, komponen, jadwal pembayaran)
4. Jam Kerja & Lembur
5. Cuti & Hari Libur
6. Ketentuan Pengakhiran Kerja (termination/PHK)
7. Klausul Non-Compete / Larangan Bekerja di Kompetitor
8. Klausul Kerahasiaan (confidentiality)
9. Penyelesaian Sengketa
10. Klausul lain yang tidak umum atau berpotensi merugikan karyawan

Ekstrak apa adanya sesuai teks kontrak, jangan menambahkan interpretasi. Jika suatu klausul tidak ditemukan dalam kontrak, tulis "Tidak disebutkan dalam kontrak"."""

ANALYSIS_PROMPT = """Anda adalah "Penasihat Kontrak Kerja" — asisten AI yang membantu fresh graduate dan pelamar kerja memahami kontrak kerja sebelum tanda tangan. Audiens Anda umumnya belum berpengalaman dan tidak punya latar belakang hukum, jadi gunakan bahasa yang sederhana, jelas, dan tidak menghakimi.

Analisis data klausul kontrak kerja yang telah diekstrak, lalu berikan output dengan format berikut:

## Skor Risiko Keseluruhan
Beri skor 1-10 (1 = sangat aman, 10 = sangat berisiko) dan level (Rendah/Sedang/Tinggi), beserta alasan singkat.

## Red Flags (jika ada)
Daftar klausul yang berpotensi merugikan karyawan. Untuk tiap red flag:
- Nama klausul
- Tingkat keparahan (Rendah/Sedang/Tinggi)
- Penjelasan sederhana kenapa ini perlu diwaspadai (hindari istilah hukum yang rumit)

## Perbandingan dengan Standar Umum
Bandingkan poin-poin penting (masa percobaan, cuti, lembur, dsb) dengan praktik ketenagakerjaan yang umum berlaku di Indonesia. Selalu sertakan disclaimer bahwa ini adalah perbandingan umum, bukan nasihat hukum resmi.

## Pertanyaan yang Sebaiknya Ditanyakan ke HR
Buat daftar pertanyaan konkret dan siap pakai yang bisa langsung ditanyakan user ke HR terkait klausul yang kurang jelas atau berisiko.

## Rekomendasi Akhir
Beri rekomendasi singkat: aman untuk ditandatangani, perlu negosiasi dulu, atau perlu konsultasi lebih lanjut dengan ahli hukum ketenagakerjaan.

Gunakan Bahasa Indonesia yang ramah dan mudah dipahami. Akhiri dengan catatan bahwa ini bukan pengganti nasihat hukum profesional."""


class ContractClauses(BaseModel):
    contract_type: str = Field(NOT_FOUND, description="Jenis kontrak kerja: PKWT, PKWTT, atau tidak disebutkan jelas")
    probation_period: str = Field(NOT_FOUND, description="Ketentuan masa percobaan (durasi, syarat, konsekuensi)")
    salary_and_benefits: str = Field(NOT_FOUND, description="Gaji pokok, tunjangan, bonus, dan jadwal pembayaran")
    working_hours_and_overtime: str = Field(NOT_FOUND, description="Jam kerja normal dan ketentuan lembur")
    leave_and_holiday: str = Field(NOT_FOUND, description="Cuti tahunan, cuti sakit, dan hari libur")
    termination_clause: str = Field(NOT_FOUND, description="Ketentuan pengakhiran kerja/PHK dari kedua pihak")
    non_compete_clause: str = Field(NOT_FOUND, description="Klausul larangan bekerja di kompetitor atau membuka usaha sejenis")
    confidentiality_clause: str = Field(NOT_FOUND, description="Klausul kerahasiaan informasi perusahaan")
    dispute_resolution: str = Field(NOT_FOUND, description="Mekanisme penyelesaian sengketa antara karyawan dan perusahaan")
    other_notable_clauses: str = Field(NOT_FOUND, description="Klausul lain yang tidak umum, ambigu, atau berpotensi merugikan karyawan")


# ---------- Stage 1: file ingestion ----------
def read_uploaded(uploaded):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        if not text.strip():
            raise ValueError("PDF tidak berisi teks (kemungkinan hasil scan). Gunakan PDF/DOCX berbasis teks.")
    elif name.endswith(".docx"):
        from docx import Document
        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Format tidak didukung. Gunakan PDF, DOCX, atau TXT.")
    if not text.strip():
        raise ValueError("Tidak ada teks yang bisa dibaca dari file tersebut.")
    return text.strip()


# ---------- Rate limiting (protects the free API quota) ----------
@st.cache_resource
def _usage():
    return {"lock": threading.Lock(), "day": date.today(), "count": 0}


def check_quota():
    u = _usage()
    with u["lock"]:
        today = date.today()
        if u["day"] != today:
            u["day"], u["count"] = today, 0
        if u["count"] >= DAILY_LIMIT:
            raise RuntimeError("Kuota demo harian sudah habis. Silakan coba lagi besok.")
        u["count"] += 1


def get_client():
    try:
        key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY belum diatur di Secrets.")
    return genai.Client(api_key=key)


# ---------- Stage 2: structured clause extraction ----------
def extract_clauses(client, text):
    resp = client.models.generate_content(
        model=EXTRACT_MODEL,
        contents=f"TEKS KONTRAK:\n\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=EXTRACT_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ContractClauses,
        ),
    )
    parsed = resp.parsed
    if isinstance(parsed, ContractClauses):
        return parsed
    return ContractClauses(**json.loads(resp.text))


# ---------- Stage 3: parser (stringify) ----------
def parse_clauses(clauses):
    return json.dumps(clauses.model_dump(), ensure_ascii=False, indent=2)


# ---------- Stage 4: risk-analysis LLM ----------
def analyze_risk(client, clauses_str):
    resp = client.models.generate_content(
        model=ANALYSIS_MODEL,
        contents=f"Data klausul kontrak yang telah diekstrak:\n\n{clauses_str}",
        config=types.GenerateContentConfig(system_instruction=ANALYSIS_PROMPT, temperature=0.07),
    )
    return resp.text


# ---------- Stage 5: UI ----------
def main():
    st.set_page_config(page_title="Kontrak Aman", page_icon="📄", layout="wide")
    st.title("📄 Kontrak Aman")
    st.caption("Pemeriksa risiko kontrak kerja untuk fresh graduate.")
    st.warning(
        "**Bukan nasihat hukum.** Alat ini hanya memberi gambaran umum. "
        "Jangan unggah kontrak asli yang berisi data pribadi; gunakan contoh fiktif atau hapus identitas terlebih dahulu."
    )

    if "contract_text" not in st.session_state:
        st.session_state.contract_text = ""

    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Unggah kontrak (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
        if st.button("Pakai contoh kontrak fiktif"):
            st.session_state.contract_text = SAMPLE_PATH.read_text(encoding="utf-8")
        st.text_area("Atau tempel teks kontrak", height=300, key="contract_text")
        run = st.button("Analisis kontrak", type="primary")

    with right:
        if not run:
            st.info("Hasil analisis akan tampil di sini.")
            return
        try:
            with st.status("Menganalisis kontrak...", expanded=True) as status:
                st.write("Tahap 1/4: membaca kontrak")
                pasted = st.session_state.contract_text.strip()
                if pasted:
                    text = pasted
                elif uploaded is not None:
                    text = read_uploaded(uploaded)
                else:
                    raise ValueError("Unggah file kontrak atau tempel teksnya terlebih dahulu.")
                if len(text) > MAX_CHARS:
                    raise ValueError(f"Kontrak terlalu panjang (maks {MAX_CHARS:,} karakter untuk demo).")
                check_quota()
                client = get_client()

                st.write("Tahap 2/4: mengekstrak klausul")
                clauses = extract_clauses(client, text)

                st.write("Tahap 3/4: menyusun data klausul")
                clauses_str = parse_clauses(clauses)

                st.write("Tahap 4/4: menganalisis risiko")
                report = analyze_risk(client, clauses_str)
                status.update(label="Selesai", state="complete", expanded=False)
        except Exception as e:
            st.error(str(e))
            return

        st.markdown(report)
        with st.expander("Klausul hasil ekstraksi (JSON)"):
            st.code(clauses_str, language="json")


if __name__ == "__main__":
    main()