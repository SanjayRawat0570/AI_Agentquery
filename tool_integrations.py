"""
PHASE 2.2: Real Tool Integrations
Integrate with external services: Stripe, SendGrid, Slack, Database, etc.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


# ==================== STRIPE INTEGRATION ====================
class StripeIntegration:
    """
    Stripe payment processing integration
    Docs: https://stripe.com/docs
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("STRIPE_API_KEY")
        self.available = bool(self.api_key)
        
        if self.available:
            try:
                import stripe
                stripe.api_key = self.api_key
                self.stripe = stripe
                logger.info("Stripe integration initialized")
            except ImportError:
                logger.warning("stripe package not installed")
                self.available = False
        else:
            logger.warning("STRIPE_API_KEY not configured")
    
    def create_payment_intent(self, amount: int, currency: str = "usd", 
                             description: str = None) -> Dict[str, Any]:
        """Create a Stripe payment intent"""
        if not self.available:
            return {"error": "Stripe not available"}
        
        try:
            intent = self.stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                description=description
            )
            return {
                "success": True,
                "payment_intent_id": intent.id,
                "client_secret": intent.client_secret,
                "amount": amount,
                "currency": currency
            }
        except Exception as e:
            logger.error(f"Stripe error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve customer from Stripe"""
        if not self.available:
            return {"error": "Stripe not available"}
        
        try:
            customer = self.stripe.Customer.retrieve(customer_id)
            return {
                "success": True,
                "customer_id": customer.id,
                "email": customer.email,
                "name": customer.name,
                "subscriptions": len(customer.subscriptions.data) if customer.subscriptions else 0
            }
        except Exception as e:
            logger.error(f"Stripe error: {e}")
            return {"success": False, "error": str(e)}
    
    def process_refund(self, payment_intent_id: str, amount: Optional[int] = None) -> Dict[str, Any]:
        """Process a refund for a payment"""
        if not self.available:
            return {"error": "Stripe not available"}
        
        try:
            refund = self.stripe.Refund.create(
                payment_intent=payment_intent_id,
                amount=amount
            )
            return {
                "success": True,
                "refund_id": refund.id,
                "status": refund.status,
                "amount": refund.amount
            }
        except Exception as e:
            logger.error(f"Stripe error: {e}")
            return {"success": False, "error": str(e)}


# ==================== SENDGRID INTEGRATION ====================
class SendGridIntegration:
    """
    SendGrid email service integration
    Docs: https://docs.sendgrid.com
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SENDGRID_API_KEY")
        self.available = bool(self.api_key)
        
        if self.available:
            try:
                from sendgrid import SendGridAPIClient
                from sendgrid.helpers.mail import Mail, Email, To, Content
                self.SendGridAPIClient = SendGridAPIClient
                self.Mail = Mail
                self.Email = Email
                self.To = To
                self.Content = Content
                self.sg = SendGridAPIClient(self.api_key)
                logger.info("SendGrid integration initialized")
            except ImportError:
                logger.warning("sendgrid package not installed")
                self.available = False
        else:
            logger.warning("SENDGRID_API_KEY not configured")
    
    def send_email(self, to_email: str, subject: str, html_content: str,
                   from_email: Optional[str] = None) -> Dict[str, Any]:
        """Send email via SendGrid"""
        if not self.available:
            return {"error": "SendGrid not available"}
        
        try:
            from_email = from_email or os.getenv("SENDGRID_FROM_EMAIL", "support@techsolutions.com")
            
            message = self.Mail(
                from_email=from_email,
                to_emails=to_email,
                subject=subject,
                html_content=html_content
            )
            
            response = self.sg.send(message)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "message_id": response.headers.get("X-Message-Id", ""),
                "to": to_email,
                "subject": subject
            }
        except Exception as e:
            logger.error(f"SendGrid error: {e}")
            return {"success": False, "error": str(e)}
    
    def send_template_email(self, to_email: str, template_id: str,
                           dynamic_template_data: Dict = None) -> Dict[str, Any]:
        """Send templated email via SendGrid"""
        if not self.available:
            return {"error": "SendGrid not available"}
        
        try:
            from sendgrid.helpers.mail import TemplateId
            from_email = os.getenv("SENDGRID_FROM_EMAIL", "support@techsolutions.com")
            
            message = self.Mail(
                from_email=from_email,
                to_emails=to_email
            )
            message.template_id = TemplateId(template_id)
            
            if dynamic_template_data:
                message.dynamic_template_data = dynamic_template_data
            
            response = self.sg.send(message)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "template_id": template_id,
                "to": to_email
            }
        except Exception as e:
            logger.error(f"SendGrid error: {e}")
            return {"success": False, "error": str(e)}


# ==================== SLACK INTEGRATION ====================
class SlackIntegration:
    """
    Slack messaging integration
    Docs: https://api.slack.com
    """
    
    def __init__(self, webhook_url: Optional[str] = None, bot_token: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.available = bool(self.webhook_url or self.bot_token)
        
        if self.available and self.bot_token:
            try:
                from slack_sdk import WebClient
                self.client = WebClient(token=self.bot_token)
                logger.info("Slack integration initialized")
            except ImportError:
                logger.warning("slack-sdk package not installed")
                self.available = False
        elif self.available:
            logger.info("Slack webhook integration initialized")
        else:
            logger.warning("SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN not configured")
    
    def send_message(self, channel: str, text: str, blocks: Optional[List] = None) -> Dict[str, Any]:
        """Send message to Slack channel"""
        if not self.available:
            return {"error": "Slack not available"}
        
        try:
            if self.bot_token and hasattr(self, 'client'):
                # Use bot token
                response = self.client.chat_postMessage(
                    channel=channel,
                    text=text,
                    blocks=blocks
                )
                return {
                    "success": True,
                    "channel": channel,
                    "timestamp": response["ts"],
                    "message_id": response["ts"]
                }
            else:
                # Use webhook
                import json
                import requests
                
                payload = {"text": text, "channel": channel}
                if blocks:
                    payload["blocks"] = blocks
                
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=5
                )
                
                return {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "channel": channel
                }
        except Exception as e:
            logger.error(f"Slack error: {e}")
            return {"success": False, "error": str(e)}
    
    def post_notification(self, title: str, message: str, severity: str = "info") -> Dict[str, Any]:
        """Post a notification to Slack (team channel)"""
        if not self.available:
            return {"error": "Slack not available"}
        
        channel = os.getenv("SLACK_NOTIFICATIONS_CHANNEL", "#alerts")
        
        color_map = {"info": "#36a64f", "warning": "#ff9900", "error": "#ff0000"}
        color = color_map.get(severity, "#36a64f")
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": title}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message}
            }
        ]
        
        return self.send_message(channel, title, blocks)


# ==================== TWILIO INTEGRATION ====================
class TwilioIntegration:
    """
    Twilio SMS and voice integration
    Docs: https://www.twilio.com/docs
    """
    
    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None):
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.available = bool(self.account_sid and self.auth_token and self.from_number)
        
        if self.available:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Twilio integration initialized")
            except ImportError:
                logger.warning("twilio package not installed")
                self.available = False
        else:
            logger.warning("Twilio credentials not configured")
    
    def send_sms(self, to_number: str, message: str) -> Dict[str, Any]:
        """Send SMS via Twilio"""
        if not self.available:
            return {"error": "Twilio not available"}
        
        try:
            msg = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            return {
                "success": True,
                "message_sid": msg.sid,
                "status": msg.status,
                "to": to_number,
                "price": str(msg.price) if msg.price else "unknown"
            }
        except Exception as e:
            logger.error(f"Twilio error: {e}")
            return {"success": False, "error": str(e)}
    
    def make_call(self, to_number: str, message: str) -> Dict[str, Any]:
        """Make a voice call via Twilio (requires TwiML)"""
        if not self.available:
            return {"error": "Twilio not available"}
        
        try:
            # Simple TwiML for text-to-speech
            twiml = f'<Response><Say>{message}</Say></Response>'
            
            call = self.client.calls.create(
                to=to_number,
                from_=self.from_number,
                twiml=twiml
            )
            
            return {
                "success": True,
                "call_sid": call.sid,
                "status": call.status,
                "to": to_number
            }
        except Exception as e:
            logger.error(f"Twilio error: {e}")
            return {"success": False, "error": str(e)}


# ==================== INTEGRATION FACTORY ====================
class ToolIntegrationFactory:
    """Factory to create and manage all tool integrations"""
    
    def __init__(self):
        self.integrations = {}
        self._initialize_integrations()
        logger.info("ToolIntegrationFactory initialized")
    
    def _initialize_integrations(self):
        """Initialize all available integrations"""
        self.integrations["stripe"] = StripeIntegration()
        self.integrations["sendgrid"] = SendGridIntegration()
        self.integrations["slack"] = SlackIntegration()
        self.integrations["twilio"] = TwilioIntegration()
    
    def get_integration(self, name: str):
        """Get an integration by name"""
        return self.integrations.get(name)
    
    def get_available_integrations(self) -> Dict[str, bool]:
        """Get status of all integrations"""
        return {
            name: integration.available
            for name, integration in self.integrations.items()
        }


# Global factory instance
integration_factory = ToolIntegrationFactory()
