from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

# Create database instance
db = SQLAlchemy()

class ProcessedImage(db.Model):
    """Model for tracking processed image data"""
    __tablename__ = 'processed_images'
    
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    input_path = db.Column(db.String(255), nullable=False)
    output_path = db.Column(db.String(255), nullable=False)
    processing_method = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # in seconds
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text, nullable=True)

    def __init__(self):
        """Initialize with default values"""
        self.original_filename = ""
        self.input_path = ""
        self.output_path = ""
        self.processing_method = ""
        self.processing_time = 0.0
        self.success = True
        self.error_message = None
        
    def __repr__(self):
        return f'<ProcessedImage {self.original_filename}>'

class ProcessedVideo(db.Model):
    """Model for tracking processed video data"""
    __tablename__ = 'processed_videos'
    
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    input_path = db.Column(db.String(255), nullable=False)
    output_path = db.Column(db.String(255), nullable=False)
    processing_method = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # in seconds
    frames_processed = db.Column(db.Integer)
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text, nullable=True)

    def __init__(self):
        """Initialize with default values"""
        self.original_filename = ""
        self.input_path = ""
        self.output_path = ""
        self.processing_method = ""
        self.processing_time = 0.0
        self.frames_processed = 0
        self.success = True
        self.error_message = None
        
    def __repr__(self):
        return f'<ProcessedVideo {self.original_filename}>'
