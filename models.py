"""
PHASE 1.3: Database Models and Integration
SQLAlchemy models for persistent storage
"""

import logging
import os
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, DateTime, Integer, JSON, Boolean, Text, create_engine
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()


class Conversation(Base):
    """Model for storing conversation data"""
    __tablename__ = "conversations"
    
    id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), nullable=True, index=True)
    agent_type = Column(String(50), nullable=True)
    turns = Column(JSON, default=list)
    conversation_metadata = Column(JSON, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    status = Column(String(20), default="active", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, customer_id={self.customer_id}, status={self.status})>"


class Customer(Base):
    """Model for storing customer data"""
    __tablename__ = "customers"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    subscription_plan = Column(String(50), nullable=True)
    account_id = Column(String(50), nullable=True, index=True)
    status = Column(String(20), default="active", index=True)
    customer_metadata = Column(JSON, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Customer(id={self.id}, name={self.name}, email={self.email})>"


class SupportTicket(Base):
    """Model for support tickets created by agents"""
    __tablename__ = "support_tickets"
    
    id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), nullable=True, index=True)
    conversation_id = Column(String(50), nullable=True, index=True)
    issue = Column(Text, nullable=False)
    priority = Column(String(20), default="medium", index=True)
    status = Column(String(20), default="open", index=True)
    assigned_agent = Column(String(50), nullable=True)
    resolution = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<SupportTicket(id={self.id}, priority={self.priority}, status={self.status})>"


class AgentAction(Base):
    """Model for tracking agent actions (tool usage, decisions)"""
    __tablename__ = "agent_actions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(50), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False, index=True)
    action_type = Column(String(50), nullable=False, index=True)  # tool_call, decision, response
    action_details = Column(JSON, default=dict)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AgentAction(id={self.id}, agent={self.agent_name}, type={self.action_type})>"


class PerformanceMetric(Base):
    """Model for storing agent performance metrics"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(50), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # response_time, accuracy, satisfaction
    metric_value = Column(JSON, nullable=False)
    conversation_id = Column(String(50), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<PerformanceMetric(agent={self.agent_name}, type={self.metric_type})>"


class DatabaseManager:
    """
    Manager for database connections and operations
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: SQLAlchemy database URL. Defaults to SQLite if not provided.
        """
        if database_url is None:
            database_url = "sqlite:///./agent_data.db"
        
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True  # Verify connections before using
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"DatabaseManager initialized with URL: {database_url}")
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def save_conversation(self, conversation_data: dict) -> bool:
        """Save or update a conversation"""
        session = self.get_session()
        try:
            conversation = session.query(Conversation).filter_by(id=conversation_data["id"]).first()
            if conversation:
                # Update existing
                for key, value in conversation_data.items():
                    setattr(conversation, key, value)
                conversation.updated_at = datetime.utcnow()
            else:
                # Create new
                conversation = Conversation(**conversation_data)
                session.add(conversation)
            
            session.commit()
            logger.info(f"Conversation saved: {conversation_data['id']}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save conversation: {e}")
            return False
        finally:
            session.close()
    
    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Retrieve a conversation by ID"""
        session = self.get_session()
        try:
            conversation = session.query(Conversation).filter_by(id=conversation_id).first()
            if conversation:
                return {
                    "id": conversation.id,
                    "customer_id": conversation.customer_id,
                    "agent_type": conversation.agent_type,
                    "turns": conversation.turns,
                    "metadata": conversation.conversation_metadata,  # Use renamed field
                    "status": conversation.status,
                    "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                    "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None
                }
            return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve conversation: {e}")
            return None
        finally:
            session.close()
    
    def log_agent_action(self, conversation_id: str, agent_name: str, action_type: str, 
                         action_details: dict, success: bool = True, error_message: Optional[str] = None):
        """Log an agent action"""
        session = self.get_session()
        try:
            action = AgentAction(
                conversation_id=conversation_id,
                agent_name=agent_name,
                action_type=action_type,
                action_details=action_details,
                success=success,
                error_message=error_message
            )
            session.add(action)
            session.commit()
            logger.debug(f"Agent action logged: {agent_name} - {action_type}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to log agent action: {e}")
            return False
        finally:
            session.close()
    
    def save_customer(self, customer_data: dict) -> bool:
        """Save or update a customer"""
        session = self.get_session()
        try:
            customer = session.query(Customer).filter_by(id=customer_data["id"]).first()
            if customer:
                for key, value in customer_data.items():
                    setattr(customer, key, value)
                customer.updated_at = datetime.utcnow()
            else:
                customer = Customer(**customer_data)
                session.add(customer)
            
            session.commit()
            logger.info(f"Customer saved: {customer_data['id']}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to save customer: {e}")
            return False
        finally:
            session.close()


# Global database manager instance (initialized lazily)
db_manager: Optional[DatabaseManager] = None


def get_db_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """Get or create global database manager instance"""
    global db_manager
    if db_manager is None:
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
        db_manager = DatabaseManager(database_url)
        db_manager.create_tables()
    return db_manager
