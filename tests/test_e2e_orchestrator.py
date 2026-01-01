#!/usr/bin/env python3
"""
Unit tests for E2E test orchestrator

Tests the core components of the E2E test without requiring actual MetaAPI connection.
"""

import sys
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from scripts.testing.test_e2e_symbols_timeframes import (
    E2ETestConfig,
    E2ETestOrchestrator,
    StageStatus,
    ActionOnError,
    StageResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    config = E2ETestConfig(
        symbols=['BTCUSD', 'EURUSD'],
        timeframes=['H1', 'H4'],
        days_history=90,
        quick_mode=True,
        max_retries=2,
        retry_delay=0.1,  # Fast retries for testing
        action_on_error=ActionOnError.RETRY,
    )
    config.results_dir = temp_dir / "results"
    config.checkpoint_dir = temp_dir / "checkpoints"
    return config


@pytest.fixture
def orchestrator(test_config):
    """Create orchestrator instance."""
    return E2ETestOrchestrator(test_config)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

def test_config_creation():
    """Test basic config creation."""
    config = E2ETestConfig(
        symbols=['BTCUSD'],
        timeframes=['H1'],
        quick_mode=False
    )
    
    assert config.symbols == ['BTCUSD']
    assert config.timeframes == ['H1']
    assert config.superpot_episodes == 80
    assert config.days_history == 365


def test_config_quick_mode():
    """Test quick mode adjusts parameters."""
    config = E2ETestConfig(
        symbols=['BTCUSD'],
        timeframes=['H1'],
        quick_mode=True
    )
    
    assert config.superpot_episodes == 30
    assert config.superpot_prune_every == 10
    assert config.days_history == 90


def test_action_on_error_enum():
    """Test ActionOnError enum values."""
    assert ActionOnError.ABORT.value == "abort"
    assert ActionOnError.SKIP.value == "skip"
    assert ActionOnError.RETRY.value == "retry"
    assert ActionOnError.PROMPT.value == "prompt"


def test_stage_status_enum():
    """Test StageStatus enum values."""
    assert StageStatus.PENDING.value == "pending"
    assert StageStatus.RUNNING.value == "running"
    assert StageStatus.COMPLETED.value == "completed"
    assert StageStatus.FAILED.value == "failed"
    assert StageStatus.SKIPPED.value == "skipped"


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================

def test_orchestrator_creation(orchestrator, test_config):
    """Test orchestrator initialization."""
    assert orchestrator.config == test_config
    assert len(orchestrator.stages) == 0
    assert orchestrator.downloaded_files == {}
    assert orchestrator.superpot_results == {}
    assert orchestrator.theorem_results == {}


def test_checkpoint_save_load(orchestrator, temp_dir):
    """Test checkpoint save and load."""
    # Add some test data
    orchestrator.downloaded_files = {"BTCUSD_H1": "/path/to/file.csv"}
    orchestrator.stages.append(StageResult(
        stage_name="Test Stage",
        status=StageStatus.COMPLETED,
        start_time=1234567890.0,
        end_time=1234567900.0,
        duration=10.0
    ))
    
    # Save checkpoint
    orchestrator.save_checkpoint()
    
    # Verify file exists
    assert orchestrator.checkpoint_file.exists()
    
    # Create new orchestrator and load
    new_orchestrator = E2ETestOrchestrator(orchestrator.config)
    success = new_orchestrator.load_checkpoint(orchestrator.checkpoint_file)
    
    assert success
    assert new_orchestrator.downloaded_files == {"BTCUSD_H1": "/path/to/file.csv"}
    assert len(new_orchestrator.stages) == 1
    assert new_orchestrator.stages[0].stage_name == "Test Stage"


def test_checkpoint_load_invalid(orchestrator, temp_dir):
    """Test loading invalid checkpoint."""
    invalid_file = temp_dir / "invalid.json"
    invalid_file.write_text("invalid json{")
    
    success = orchestrator.load_checkpoint(invalid_file)
    assert not success


def test_handle_error_abort(orchestrator):
    """Test error handling with ABORT action."""
    orchestrator.config.action_on_error = ActionOnError.ABORT
    
    error = ValueError("Test error")
    action = orchestrator.handle_error("Test Stage", error)
    
    assert action == ActionOnError.ABORT


def test_handle_error_skip(orchestrator):
    """Test error handling with SKIP action."""
    orchestrator.config.action_on_error = ActionOnError.SKIP
    
    error = ValueError("Test error")
    action = orchestrator.handle_error("Test Stage", error)
    
    assert action == ActionOnError.SKIP


def test_handle_error_retry(orchestrator):
    """Test error handling with RETRY action."""
    orchestrator.config.action_on_error = ActionOnError.RETRY
    
    error = ValueError("Test error")
    action = orchestrator.handle_error("Test Stage", error)
    
    assert action == ActionOnError.RETRY


# =============================================================================
# STAGE EXECUTION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_run_stage_success(orchestrator):
    """Test successful stage execution."""
    def test_stage():
        return {"status": "success"}
    
    success, result = await orchestrator.run_stage("Test Stage", test_stage)
    
    assert success
    assert result == {"status": "success"}
    assert len(orchestrator.stages) == 1
    assert orchestrator.stages[0].status == StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_stage_async_success(orchestrator):
    """Test successful async stage execution."""
    async def test_stage():
        await asyncio.sleep(0.01)
        return {"status": "async_success"}
    
    success, result = await orchestrator.run_stage("Async Test Stage", test_stage)
    
    assert success
    assert result == {"status": "async_success"}
    assert orchestrator.stages[0].status == StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_stage_failure_retry_succeed(orchestrator):
    """Test stage that fails once then succeeds on retry."""
    call_count = 0
    
    def test_stage():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("First attempt fails")
        return {"status": "success_on_retry"}
    
    orchestrator.config.action_on_error = ActionOnError.RETRY
    success, result = await orchestrator.run_stage("Retry Stage", test_stage)
    
    assert success
    assert result == {"status": "success_on_retry"}
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_stage_failure_max_retries(orchestrator):
    """Test stage that fails with max retries exceeded."""
    def test_stage():
        raise ValueError("Always fails")
    
    orchestrator.config.action_on_error = ActionOnError.RETRY
    orchestrator.config.max_retries = 1
    
    success, result = await orchestrator.run_stage("Failing Stage", test_stage)
    
    assert not success
    assert result is None
    assert orchestrator.stages[0].status == StageStatus.FAILED


@pytest.mark.asyncio
async def test_run_stage_skip_on_error(orchestrator):
    """Test stage that is skipped on error."""
    def test_stage():
        raise ValueError("Error to skip")
    
    orchestrator.config.action_on_error = ActionOnError.SKIP
    
    success, result = await orchestrator.run_stage("Skipped Stage", test_stage, critical=False)
    
    assert success  # Returns success when skipped
    assert result is None
    assert orchestrator.stages[0].status == StageStatus.SKIPPED


@pytest.mark.asyncio
async def test_run_stage_abort_on_critical(orchestrator):
    """Test critical stage aborts on error."""
    def test_stage():
        raise ValueError("Critical error")
    
    orchestrator.config.action_on_error = ActionOnError.RETRY
    orchestrator.config.max_retries = 1
    
    success, result = await orchestrator.run_stage("Critical Stage", test_stage, critical=True)
    
    assert not success
    assert orchestrator.stages[0].status == StageStatus.FAILED


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

def test_stage_validate_data_success(orchestrator, temp_dir):
    """Test data validation with valid data."""
    import pandas as pd
    
    # Create test data file
    data_file = temp_dir / "BTCUSD_H1.csv"
    df = pd.DataFrame({
        '<DATE>': ['2024.01.01', '2024.01.01'],
        '<TIME>': ['00:00:00', '01:00:00'],
        '<OPEN>': [45000.0, 45100.0],
        '<HIGH>': [45200.0, 45300.0],
        '<LOW>': [44900.0, 45000.0],
        '<CLOSE>': [45100.0, 45200.0],
        '<TICKVOL>': [100, 110],
    })
    df.to_csv(data_file, sep='\t', index=False)
    
    # Set up orchestrator with test file
    orchestrator.downloaded_files = {"BTCUSD_H1": str(data_file)}
    
    # Run validation
    result = orchestrator.stage_validate_data()
    
    assert result['BTCUSD_H1']['valid']
    assert result['BTCUSD_H1']['rows'] == 2
    assert result['BTCUSD_H1']['nulls'] == 0


def test_stage_validate_data_missing_columns(orchestrator, temp_dir):
    """Test data validation with missing columns."""
    import pandas as pd
    
    # Create test data file with missing columns
    data_file = temp_dir / "BTCUSD_H1.csv"
    df = pd.DataFrame({
        '<DATE>': ['2024.01.01'],
        '<TIME>': ['00:00:00'],
        '<OPEN>': [45000.0],
    })
    df.to_csv(data_file, sep='\t', index=False)
    
    orchestrator.downloaded_files = {"BTCUSD_H1": str(data_file)}
    
    # Should raise ValueError for missing columns
    with pytest.raises(ValueError, match="Missing columns"):
        orchestrator.stage_validate_data()


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================

def test_generate_summary_empty(orchestrator):
    """Test summary generation with no stages."""
    summary = orchestrator._generate_summary()
    
    assert summary['total_stages'] == 0
    assert summary['completed'] == 0
    assert summary['failed'] == 0
    assert summary['success_rate'] == 0


def test_generate_summary_with_stages(orchestrator):
    """Test summary generation with stages."""
    orchestrator.stages = [
        StageResult("Stage1", StageStatus.COMPLETED, 1.0, 2.0, 1.0),
        StageResult("Stage2", StageStatus.COMPLETED, 2.0, 4.0, 2.0),
        StageResult("Stage3", StageStatus.FAILED, 4.0, 5.0, 1.0),
    ]
    
    summary = orchestrator._generate_summary()
    
    assert summary['total_stages'] == 3
    assert summary['completed'] == 2
    assert summary['failed'] == 1
    assert summary['success_rate'] == pytest.approx(66.67, rel=0.1)
    assert summary['total_duration_sec'] == 4.0


def test_generate_html_report(orchestrator):
    """Test HTML report generation."""
    report = {
        'test_id': 'test_123',
        'timestamp': '2024-01-01T00:00:00',
        'config': {
            'symbols': ['BTCUSD'],
            'timeframes': ['H1'],
            'days_history': 90,
            'quick_mode': True,
        },
        'stages': [
            {
                'stage_name': 'Test Stage',
                'status': 'completed',
                'duration': 10.0,
                'retry_count': 0,
            }
        ],
        'summary': {
            'total_stages': 1,
            'completed': 1,
            'failed': 0,
            'success_rate': 100.0,
            'total_duration_sec': 10.0,
        }
    }
    
    html = orchestrator._generate_html_report(report)
    
    assert '<!DOCTYPE html>' in html
    assert 'E2E Test Report' in html
    assert 'test_123' in html
    assert 'BTCUSD' in html
    assert '100.0%' in html


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_full_workflow_mock(orchestrator):
    """Test full workflow with mocked stages."""
    # Mock all stage functions to avoid actual execution
    async def mock_auth_stage():
        return {'account_id': 'test', 'account_name': 'Test Account'}
    
    def mock_validate_stage():
        return {'BTCUSD_H1': {'valid': True, 'rows': 100}}
    
    def mock_superpot_stage():
        raise ImportError("Superpot not available")
    
    def mock_theorem_stage():
        raise ImportError("Theorems not available")
    
    def mock_report_stage():
        return {'json_report': 'test.json', 'html_report': 'test.html'}
    
    # Run individual stages with mocks
    success1, _ = await orchestrator.run_stage("MetaAPI Auth", mock_auth_stage)
    assert success1
    
    # Set downloaded files manually for validation stage
    orchestrator.downloaded_files = {'BTCUSD_H1': '/tmp/test.csv'}
    
    success2, _ = await orchestrator.run_stage("Data Validation", mock_validate_stage)
    assert success2
    
    # Skippable stages - set action to skip so they continue
    orchestrator.config.action_on_error = ActionOnError.SKIP
    
    success3, _ = await orchestrator.run_stage("Superpot", mock_superpot_stage, critical=False)
    assert success3  # Skipped counts as success
    
    success4, _ = await orchestrator.run_stage("Theorems", mock_theorem_stage, critical=False)
    assert success4  # Skipped counts as success
    
    success5, _ = await orchestrator.run_stage("Report", mock_report_stage, critical=False)
    assert success5
    
    # Verify stages were recorded
    assert len(orchestrator.stages) == 5
    assert orchestrator.stages[0].status == StageStatus.COMPLETED
    assert orchestrator.stages[1].status == StageStatus.COMPLETED
    assert orchestrator.stages[2].status == StageStatus.SKIPPED
    assert orchestrator.stages[3].status == StageStatus.SKIPPED
    assert orchestrator.stages[4].status == StageStatus.COMPLETED


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
