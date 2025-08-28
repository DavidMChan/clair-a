from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import trange
from transformers import AutoTokenizer
from transformers import logging as trf_logging

from .data import infer_preprocess
from .model import BERTFlatClassifier

PRETRAIN_ECHECKERS = {
    'echecker_clotho_audiocaps_base': ("https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_base.ckpt", "1a719f090af70614bbdb9f9437530b7e133c48cfa4a58d964de0d47fc974a2fa"),
    'echecker_clotho_audiocaps_tiny': ("https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_tiny.ckpt", "90ed0ac5033ec497ec66d4f68588053813e085671136dae312097c96c504f673"),
    "none": (None, None)
}

def load_pretrain_echecker(echecker_model: str, device: str = 'cuda', use_proxy: bool = False, proxies: Optional[Dict] = None) -> BERTFlatClassifier:
    """
    Load a pretrained error checker model.
    
    Args:
        echecker_model: Name of the pretrained model
        device: Device to load the model on
        use_proxy: Whether to use proxy for downloading
        proxies: Proxy configuration
        
    Returns:
        Loaded BERTFlatClassifier model
    """
    from .download_utils import RemoteFileMetadata, check_download_resource
    
    trf_logging.set_verbosity_error()  # suppress loading warnings
    url, checksum = PRETRAIN_ECHECKERS[echecker_model]
    remote = RemoteFileMetadata(
        filename=f'{echecker_model}.ckpt',
        url=url,
        checksum=checksum
    )
    file_path = check_download_resource(remote, use_proxy, proxies)
    
    # Load model with updated error handling
    try:
        model_states = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading model states: {e}")
        # Fallback: try loading with weights_only=True for newer PyTorch versions
        try:
            model_states = torch.load(file_path, map_location='cpu', weights_only=True)
        except:
            model_states = torch.load(file_path, map_location='cpu')
    
    clf = BERTFlatClassifier(
        model_type=model_states['model_type'], 
        num_classes=model_states['num_classes']
    )
    
    # Use our custom load_state_dict method that handles the position_ids issue
    clf.load_state_dict(model_states['state_dict'])
    clf.eval()
    clf.to(device)
    return clf


class Evaluator:
    """
    FENSE evaluator for fluency and semantic similarity evaluation.
    """
    
    def __init__(
        self, 
        batch_size: int = 32, 
        device: str = 'cuda', 
        sbert_model: str = "paraphrase-TinyBERT-L6-v2", 
        echecker_model: str = "echecker_clotho_audiocaps_base", 
        error_threshold: float = 0.9, 
        penalty: float = 0.9, 
        use_proxy: bool = False, 
        proxies: Optional[Dict] = None
    ):
        """
        Initialize the FENSE evaluator.
        
        Args:
            batch_size: Batch size for processing
            device: Device to run models on ('cuda' or 'cpu')
            sbert_model: Sentence transformer model name
            echecker_model: Error checker model name
            error_threshold: Threshold for error detection
            penalty: Penalty factor for detected errors
            use_proxy: Whether to use proxy for downloads
            proxies: Proxy configuration
        """
        assert echecker_model in PRETRAIN_ECHECKERS, f"Unknown echecker_model: {echecker_model}"
        
        self.batch_size = batch_size
        self.device = device
        self.sbert_model = sbert_model
        self.echecker_model = echecker_model
        self.error_threshold = error_threshold
        self.penalty = penalty

        # Initialize sentence transformer
        self.sbert = SentenceTransformer(sbert_model, device=device)
        
        # Initialize error checker if not "none"
        if echecker_model != "none":
            self.echecker = load_pretrain_echecker(echecker_model, device, use_proxy, proxies)
            self.echecker_tokenizer = AutoTokenizer.from_pretrained(self.echecker.model_type)
            self.echecker.to(device)
            self.echecker.eval()

    def encode_sents_sbert(self, sents: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode multiple sentences using SBERT."""
        return self.sbert.encode(
            sents, 
            convert_to_tensor=True, 
            normalize_embeddings=True, 
            batch_size=batch_size, 
            show_progress_bar=True
        )
    
    @lru_cache(maxsize=32)   # reuse cache if encode the same sentence
    def encode_sent_sbert(self, sent: str) -> torch.Tensor:
        """Encode a single sentence using SBERT."""
        return self.sbert.encode(sent, convert_to_tensor=True, normalize_embeddings=True)

    def detect_error_sents(self, sents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Detect errors in multiple sentences.
        
        Args:
            sents: List of sentences to check
            batch_size: Batch size for processing
            
        Returns:
            Array of error indicators (1 for error, 0 for no error)
        """
        if self.echecker_model == "none":
            return np.zeros(len(sents))
            
        if len(sents) <= batch_size:
            batch = infer_preprocess(self.echecker_tokenizer, sents, max_len=64)
            for k, v in batch.items():
                batch[k] = v.to(self.device)
            with torch.no_grad():
                logits = self.echecker(**batch)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            probs = []
            for i in trange(0, len(sents), batch_size):
                batch = infer_preprocess(self.echecker_tokenizer, sents[i:i+batch_size], max_len=64)
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                with torch.no_grad():
                    batch_logits = self.echecker(**batch)
                    batch_probs = torch.sigmoid(batch_logits).detach().cpu().numpy()
                    # Handle the case where we have multiple classes
                    if batch_probs.ndim > 1:
                        batch_probs = batch_probs[:, -1]  # Take the last class (error class)
                probs.append(batch_probs)
            probs = np.concatenate(probs)
        
        return (probs > self.error_threshold).astype(float)

    @lru_cache(maxsize=32)   # reuse cache if infer with the same sentence
    def detect_error_sent(self, sent: str, return_error_prob: bool = False) -> Union[bool, Tuple[bool, float]]:
        """
        Detect errors in a single sentence.
        
        Args:
            sent: Sentence to check
            return_error_prob: Whether to return error probability
            
        Returns:
            Error indicator or tuple of (error_indicator, error_probability)
        """
        if self.echecker_model == "none":
            if return_error_prob:
                return False, 0.0
            return False
            
        batch = infer_preprocess(self.echecker_tokenizer, [sent], max_len=64)
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        with torch.no_grad():
            logits = self.echecker(**batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        
        # Handle the case where we have multiple classes
        if probs.ndim > 1 and probs.shape[1] > 1:
            error_prob = probs[0, -1]  # Take the last class (error class)
        else:
            error_prob = probs[0] if probs.ndim == 1 else probs[0, 0]
            
        has_error = error_prob > self.error_threshold
        
        if return_error_prob:
            return has_error, error_prob
        else:
            return has_error 

    def corpus_score(
        self, 
        cands: List[str], 
        list_refs: List[List[str]], 
        agg_score: str = 'mean'
    ) -> Union[float, List[float]]:
        """
        Compute corpus-level scores.
        
        Args:
            cands: List of candidate sentences
            list_refs: List of reference sentence lists
            agg_score: Aggregation method ('mean', 'max', 'none')
            
        Returns:
            Aggregated score or list of scores
        """
        assert len(cands) == len(list_refs)
        assert agg_score in {'none', 'mean', 'max'}
        
        rng_ids = [0]
        all_refs = []
        for lst in list_refs:
            rng_ids.append(rng_ids[-1] + len(lst))
            all_refs.extend(lst)
        
        print("Encoding sentences")
        emb_cands = self.encode_sents_sbert(cands, self.batch_size)
        emb_refs = self.encode_sents_sbert(all_refs, self.batch_size)
        
        sim_scores = [
            (emb_cands[i] @ emb_refs[rng_ids[i]:rng_ids[i+1]].T).mean().detach().cpu().item()
            for i in range(len(cands))
        ]
        
        if self.echecker_model == "none":
            if agg_score == 'mean':
                return np.mean(sim_scores)
            elif agg_score == 'max':
                return np.max(sim_scores)
            else:
                return sim_scores
        else:
            sim_scores = np.array(sim_scores)
            print("Performing error detection")
            has_error = self.detect_error_sents(cands, self.batch_size)
            penalized_scores = sim_scores * (1 - self.penalty * has_error)
            if agg_score == 'mean':
                return np.mean(penalized_scores)
            elif agg_score == 'max':
                return np.max(penalized_scores)
            else:
                return penalized_scores.tolist()

    def sentence_score(
        self, 
        cand: str, 
        refs: List[str], 
        return_error_prob: bool = False
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Compute sentence-level score.
        
        Args:
            cand: Candidate sentence
            refs: List of reference sentences
            return_error_prob: Whether to return detailed error information
            
        Returns:
            Score or tuple of (score, error_prob, penalized_score)
        """
        emb_cand = self.encode_sent_sbert(cand)
        emb_refs = self.encode_sents_sbert(refs, self.batch_size)
        scores = emb_cand @ emb_refs.T
        
        if self.echecker_model == "none":
            return scores.mean().detach().cpu().item()
        else:
            score = scores.mean().detach().cpu().item()
            if not return_error_prob:
                has_error = self.detect_error_sent(cand)
                penalized_score = (1 - self.penalty) * score if has_error else score
                return penalized_score
            else:
                has_error, error_prob = self.detect_error_sent(cand, return_error_prob=True)
                penalized_score = (1 - self.penalty) * score if has_error else score
                return score, error_prob, penalized_score


if __name__ == "__main__":
    evaluator = Evaluator(
        device='cpu', 
        sbert_model='paraphrase-MiniLM-L6-v2', 
        echecker_model='echecker_clotho_audiocaps_tiny'
    )
