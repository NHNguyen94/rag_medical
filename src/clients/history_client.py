import io
from datetime import datetime
from typing import Dict

import pandas as pd
import requests
from fpdf import FPDF

from src.utils.helpers import sanitize_text


class HistoryClient:
    def __init__(self, base_url: str, api_version: str = "v1"):
        self.api_url = f"{base_url.rstrip('/')}/{api_version}/history"

    def delete_chat_history(self, user_id: str) -> Dict:
        endpoint = f"{self.api_url}/delete_chat_history"
        payload = {"user_id": user_id}

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def get_chat_history(self, user_id: str, limit: int = 10):
        response = requests.get(
            f"{self.api_url}/chat-history/{user_id}", params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def delete_single_chat_message(self, chat_id: str):
        response = requests.delete(f"{self.api_url}/chat-history/message/{chat_id}")
        response.raise_for_status()
        return response.json()


class ChatPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.export_time = datetime.now().strftime("%B %d, %Y %H:%M")

    def footer(self):
        # This method is automatically called by FPDF for every page
        self.set_y(-20)
        self.set_font("Arial", "I", 8)
        self.cell(
            0,
            5,
            f"Exported on: {self.export_time}   |   Page {self.page_no()}",
            align="C",
        )


class ChatExportManager:
    def __init__(self):
        pass

    def generate_chat_dataframe(self, history: list) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Message": chat["message"],
                    "Response": chat["response"],
                    "Timestamp": datetime.fromisoformat(chat["created_at"]).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
                for chat in history
            ]
        )

    def export_to_excel(self, df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(
                writer, index=False, sheet_name="Chat History", engine="openpyxl"
            )
        return output.getvalue()

    def export_to_pdf(self, df: pd.DataFrame) -> io.BytesIO:
        pdf = ChatPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()

        # Header
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Chat History", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.ln(5)

        for idx, row in df.iterrows():
            # Check if we need a manual page break (optional, since auto page break handles most cases)
            if pdf.get_y() > 240:  # Leave more room for footer
                pdf.add_page()

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Message #{idx + 1}", ln=True)

            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 10, f"Time: {row['Timestamp']}", ln=True)
            # pdf.multi_cell(0, 10, f"You: {row['Message']}")
            # pdf.multi_cell(0, 10, f"Bot: {row['Response']}")
            pdf.multi_cell(0, 10, f"You: {sanitize_text(row['Message'])}")
            pdf.multi_cell(0, 10, f"Bot: {sanitize_text(row['Response'])}")
            pdf.cell(0, 10, "-" * 60, ln=True)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        return io.BytesIO(pdf_bytes)
