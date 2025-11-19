# email_notifier.py
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from typing import Optional


def send_email(subject: str, body: str) -> bool:
    """Send an email notification with the provided subject and body.
    
    Returns True if successful, False otherwise.
    """
    load_dotenv()
    
    # Get email configuration from .env
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    recipient_email = os.getenv('RECIPIENT_EMAIL')
    email_subject_prefix = os.getenv('EMAIL_SUBJECT_PREFIX', '')
    
    if not all([sender_email, sender_password, recipient_email]):
        print("Error: Email configuration missing in .env file")
        print("Required: SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL")
        return False
    
    # Add prefix to subject if provided
    if email_subject_prefix:
        subject = f"{email_subject_prefix} {subject}"
    
    try:
        # Create message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject
        
        message.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server and send
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        print(f"Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def send_new_match_notification(matches: int, fitness: float, expression: str, run: int) -> bool:
    """Send notification about a new best match found."""
    subject = f"🎯 New Best Match Found: {matches} primes!"
    
    body = f"""
New Best Match Found!
=====================

Run: #{run}
Matches: {matches} primes
Fitness: {fitness:.4f}
Expression: {expression}

This is a new unique best result that has been added to seeds.csv.
"""
    
    return send_email(subject, body)