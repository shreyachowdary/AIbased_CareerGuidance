"""
Email service for registration thank-you emails.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER or "noreply@careerpath.ai")


def send_registration_email(to_email: str, name: str) -> Tuple[bool, str]:
    """
    Send thank-you email after registration.
    Returns (success, message).
    """
    if not SMTP_USER or not SMTP_PASS:
        return False, "Email not configured. Set SMTP_USER and SMTP_PASS in .env"

    subject = "Welcome to CareerPath AI!"
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <h2 style="color: #6366f1;">Thank you for registering!</h2>
        <p>Hi {name},</p>
        <p>Welcome to <strong>CareerPath AI</strong> — your intelligent career companion.</p>
        <p>You're all set to:</p>
        <ul>
            <li>Upload your resume and get matched to jobs</li>
            <li>Identify skill gaps and get course recommendations</li>
            <li>Follow a personalized learning roadmap</li>
        </ul>
        <p>Get started by uploading your resume and running your career analysis.</p>
        <p>Best of luck on your career journey!</p>
        <p>— The CareerPath AI Team</p>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL, to_email, msg.as_string())
        return True, "Welcome email sent!"
    except Exception as e:
        return False, str(e)
