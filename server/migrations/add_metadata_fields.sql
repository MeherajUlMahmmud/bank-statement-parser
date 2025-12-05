-- Migration to add metadata fields to bank_statements table
-- Run this if you have existing data

ALTER TABLE bank_statements ADD COLUMN raw_extraction_data JSON;
ALTER TABLE bank_statements ADD COLUMN token_usage JSON;
ALTER TABLE bank_statements ADD COLUMN confidence_scores JSON;
ALTER TABLE bank_statements ADD COLUMN processing_logs JSON DEFAULT '[]';
ALTER TABLE bank_statements ADD COLUMN validation_errors JSON;
