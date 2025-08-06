import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from typing import List, Optional

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class BM25IndexCleaner:
    def __init__(self, index_dir: str):
        """
        Initialize the BM25 index cleaner
        
        Args:
            index_dir: Directory containing BM25 index files
        """
        self.index_dir = Path(index_dir)
        logger.info(f"Initialized BM25 index cleaner for directory: {self.index_dir}")
    
    def list_index_files(self) -> List[Path]:
        """
        List all BM25 index files in the directory
        
        Returns:
            List of Path objects for index files
        """
        if not self.index_dir.exists():
            logger.warning(f"Index directory does not exist: {self.index_dir}")
            return []
        
        # Get all .pkl files
        index_files = list(self.index_dir.glob("*.pkl"))
        logger.info(f"Found {len(index_files)} index files")
        return index_files
    
    def get_index_info(self, index_file: Path) -> dict:
        """
        Get information about an index file
        
        Args:
            index_file: Path to index file
            
        Returns:
            Dictionary containing index information
        """
        try:
            stats = index_file.stat()
            
            # Try to load the index to get document count
            try:
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                doc_count = len(index_data.get('documents', {}))
            except Exception as e:
                logger.warning(f"Could not read document count from {index_file}: {e}")
                doc_count = None
            
            return {
                'name': index_file.name,
                'size_bytes': stats.st_size,
                'size_mb': stats.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stats.st_ctime),
                'modified': datetime.fromtimestamp(stats.st_mtime),
                'document_count': doc_count
            }
        except Exception as e:
            logger.error(f"Error getting info for {index_file}: {e}")
            return {}
    
    def delete_index(self, index_file: Path) -> bool:
        """
        Delete an index file
        
        Args:
            index_file: Path to index file
            
        Returns:
            True if deletion was successful
        """
        try:
            # Delete the main index file
            index_file.unlink()
            
            # Delete any temporary files
            temp_file = Path(f"{index_file}.temp")
            if temp_file.exists():
                temp_file.unlink()
            
            logger.info(f"Successfully deleted index file: {index_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting index file {index_file}: {e}")
            return False
    
    def cleanup_old_indices(self, days_old: int = 30) -> int:
        """
        Delete index files older than specified days
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for index_file in self.list_index_files():
            try:
                info = self.get_index_info(index_file)
                if info.get('modified') and info['modified'] < cutoff_date:
                    if self.delete_index(index_file):
                        deleted_count += 1
            except Exception as e:
                logger.error(f"Error processing {index_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old index files")
        return deleted_count
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary index files
        
        Returns:
            Number of temporary files deleted
        """
        temp_files = list(self.index_dir.glob("*.temp"))
        deleted_count = 0
        
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} temporary files")
        return deleted_count
    
    def backup_index(self, backup_dir: Optional[str] = None) -> bool:
        """
        Backup the index directory
        
        Args:
            backup_dir: Directory to store backup. If None, creates 'backup' subdirectory
            
        Returns:
            True if backup was successful
        """
        try:
            # Set up backup directory
            if backup_dir is None:
                backup_dir = self.index_dir.parent / 'backup'
            else:
                backup_dir = Path(backup_dir)
            
            # Create backup directory if it doesn't exist
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamp for backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"bm25_index_backup_{timestamp}"
            
            # Copy index directory to backup
            shutil.copytree(self.index_dir, backup_path)
            
            logger.info(f"Successfully backed up index to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up index: {e}")
            return False

def main():
    """Example usage of the BM25IndexCleaner"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize cleaner
    index_dir = "/home/vishwa/harry_rag/intuitiveobjects-rag-api/bm25_index_store"
    cleaner = BM25IndexCleaner(index_dir)
    
    # Backup current index
    logger.info("Backing up current index...")
    cleaner.backup_index()
    
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    temp_count = cleaner.cleanup_temp_files()
    
    # List all indices with info
    logger.info("\nCurrent index files:")
    for index_file in cleaner.list_index_files():
        info = cleaner.get_index_info(index_file)
        logger.info(f"\nFile: {info['name']}")
        logger.info(f"Size: {info['size_mb']:.2f} MB")
        logger.info(f"Modified: {info['modified']}")
        logger.info(f"Documents: {info['document_count']}")
    
    # Clean up old indices
    logger.info("\nCleaning up old indices...")
    old_count = cleaner.cleanup_old_indices(days_old=0)
    
    logger.info(f"\nCleanup Summary:")
    logger.info(f"Temporary files removed: {temp_count}")
    logger.info(f"Old index files removed: {old_count}")

if __name__ == "__main__":
    main()
