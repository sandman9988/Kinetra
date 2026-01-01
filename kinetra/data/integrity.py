"""
Data Integrity Management
=========================

Data validation, gap detection, and quality scoring.
Consolidated from data_manager.py and unified_data_manager.py.

Features:
- Gap detection
- Data quality scoring
- Checksum verification
- Integrity reports
"""

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class DataIntegrity:
    """Data integrity information."""
    file_path: str
    checksum: str
    bars_count: int
    start_time: str
    end_time: str
    gaps_count: int
    quality_score: float  # 0-1
    validation_date: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataGap:
    """Represents a gap in data."""
    start: datetime
    end: datetime
    expected_bars: int
    gap_hours: float


class IntegrityChecker:
    """
    Data integrity checking and validation.
    
    Features:
    - Gap detection with weekend awareness
    - Quality scoring
    - Checksum verification
    """
    
    def __init__(self):
        """Initialize integrity checker."""
        # Timeframe intervals in minutes
        self.intervals = {
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }
        
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
                
        return sha256.hexdigest()
        
    def detect_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str,
        time_column: str = 'time'
    ) -> List[DataGap]:
        """
        Detect gaps in time series data.
        
        Args:
            df: DataFrame with time column
            timeframe: Timeframe (e.g., 'H1', 'M15')
            time_column: Name of time column
            
        Returns:
            List of detected gaps
        """
        if timeframe not in self.intervals:
            return []
            
        expected_interval = timedelta(minutes=self.intervals[timeframe])
        threshold = expected_interval * 3  # Allow 3x for weekend gaps
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
            
        df = df.sort_values(time_column)
        
        gaps = []
        for i in range(1, len(df)):
            prev_time = df.iloc[i - 1][time_column]
            curr_time = df.iloc[i][time_column]
            
            gap = curr_time - prev_time
            
            # Check if gap exceeds threshold
            if gap > threshold:
                # Check if it's just a weekend
                is_weekend = (
                    prev_time.weekday() == 4 and  # Friday
                    curr_time.weekday() == 0 and  # Monday
                    gap <= timedelta(days=3)
                )
                
                if not is_weekend:
                    expected_bars = int(gap / expected_interval)
                    gaps.append(DataGap(
                        start=prev_time,
                        end=curr_time,
                        expected_bars=expected_bars,
                        gap_hours=gap.total_seconds() / 3600
                    ))
                    
        return gaps
        
    def calculate_quality_score(
        self,
        df: pd.DataFrame,
        timeframe: str,
        gaps: List[DataGap]
    ) -> float:
        """
        Calculate data quality score (0-1).
        
        Factors:
        - Number of gaps
        - Missing bars percentage
        - Data completeness
        
        Args:
            df: DataFrame
            timeframe: Timeframe
            gaps: Detected gaps
            
        Returns:
            Quality score (1.0 = perfect)
        """
        if len(df) == 0:
            return 0.0
            
        # Calculate missing bars
        total_missing = sum(g.expected_bars for g in gaps)
        total_bars = len(df) + total_missing
        
        # Completeness score
        completeness = len(df) / total_bars if total_bars > 0 else 0.0
        
        # Gap penalty (fewer gaps = better)
        gap_penalty = min(len(gaps) * 0.05, 0.3)  # Max 30% penalty
        
        # Final score
        score = max(0.0, completeness - gap_penalty)
        
        return score
        
    def check_file(
        self,
        file_path: Path,
        timeframe: str,
        time_column: str = 'time'
    ) -> DataIntegrity:
        """
        Check integrity of data file.
        
        Args:
            file_path: Path to data file
            timeframe: Timeframe
            time_column: Name of time column
            
        Returns:
            DataIntegrity report
        """
        # Calculate checksum
        checksum = self.calculate_checksum(file_path)
        
        # Load data
        df = pd.read_csv(file_path)
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Detect gaps
        gaps = self.detect_gaps(df, timeframe, time_column)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(df, timeframe, gaps)
        
        return DataIntegrity(
            file_path=str(file_path),
            checksum=checksum,
            bars_count=len(df),
            start_time=df[time_column].min().isoformat(),
            end_time=df[time_column].max().isoformat(),
            gaps_count=len(gaps),
            quality_score=quality_score,
            validation_date=datetime.now().isoformat()
        )
