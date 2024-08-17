
    
from typing import List, Dict, Any, Optional
from datasets import Dataset as HFDataset, load_from_disk, load_dataset
from torch.utils.data import Dataset
import torch as t
from tqdm import tqdm
import numpy as np
import os

from src.utils.mech_interp_othello_utils_adopted import (
    to_string,
    OthelloBoardState
)

def state_stack_to_one_hot(state_stack):
    '''
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    '''
    one_hot = t.zeros(
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        8,
        8,
        3, # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0 
    one_hot[..., 1] = state_stack == -1 
    one_hot[..., 2] = state_stack == 1 

    return one_hot

class OthelloDataset(Dataset):
    def __init__(self, split: Optional[str], seq_len: int, max_samples: Optional[int] = None, process=False):
        self.seq_len = seq_len
        self.max_samples = max_samples
        self._data: List[Dict[str, Any]] = []
        self.games_int = None
        self.games_string = None
        self.states = None
        self.valid_moves = None
        self.flipped_states = None
        self.states_flipped_one_hot = None
        self.states_flipped_value = None

        if split is not None:
            self._load_data(split)
            self._convert_tokens_to_tensors()
            self._convert_tokens_to_strings()
            
            if process:
                self._process_data()
        
    def _load_data(self, split: str) -> None:
        dataset = load_dataset("taufeeque/othellogpt", split=split, streaming=True)
        for i, item in enumerate(tqdm(dataset, desc="Loading and processing dataset")):
            if self.max_samples is not None and i >= self.max_samples:
                break
            tokens = item["tokens"][:self.seq_len]
            self._data.append({"tokens": tokens})

    def _process_data(self) -> None:
        self._generate_states_and_valid_moves()
        self._generate_flipped_states_and_values()

    def _convert_tokens_to_tensors(self) -> None:
        self.games_int = t.tensor([game["tokens"] for game in self._data])

    def _convert_tokens_to_strings(self) -> None:
        self.games_string = t.tensor(to_string(self.games_int))

    def _generate_states_and_valid_moves(self) -> None:
        def one_hot(list_of_ints, num_classes=64):
            out = t.zeros((num_classes,), dtype=t.float32)
            out[list_of_ints] = 1.
            return out

        self.states = np.zeros((self.max_samples, 60, 8, 8), dtype=np.float32)
        self.valid_moves = t.zeros((self.max_samples, 60, 64), dtype=t.float32)

        for i in tqdm(range(self.max_samples), desc="Generating states and valid moves"):
            board = OthelloBoardState()
            for j in range(59):
                board.umpire(self.games_string[i, j].item())
                self.states[i, j] = board.state
                self.valid_moves[i, j] = one_hot(board.get_valid_moves())

    def _generate_flipped_states_and_values(self) -> None:
        alternating = np.array([-1 if i % 2 == 0 else 1 for i in range(self.games_int.shape[1])])
        self.flipped_states = self.states * alternating[None, :, None, None]
        self.states_flipped_one_hot = state_stack_to_one_hot(t.tensor(self.flipped_states))
        self.states_flipped_value = self.states_flipped_one_hot.argmax(dim=-1)

        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.states_flipped_value = self.states_flipped_value.to(device)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, t.Tensor]:
        item = self._data[idx]
        tokens = item["tokens"]
        # return tokens
        return {'tokens': t.tensor(tokens, dtype=t.long)}

    def save(self, path: str) -> None:
        hf_dataset = HFDataset.from_dict({"tokens": [item["tokens"] for item in self._data]})
        hf_dataset.save_to_disk(path)

        # Save additional tensors
        tensor_path = f"{path}/tensors"
        os.makedirs(tensor_path, exist_ok=True)
        t.save(self.games_int, os.path.join(tensor_path, "games_int.pt"))
        t.save(self.games_string, os.path.join(tensor_path, "games_string.pt"))
        np.save(os.path.join(tensor_path, "states.npy"), self.states)
        t.save(self.valid_moves, os.path.join(tensor_path, "valid_moves.pt"))
        np.save(os.path.join(tensor_path, "flipped_states.npy"), self.flipped_states)
        t.save(self.states_flipped_one_hot, os.path.join(tensor_path, "states_flipped_one_hot.pt"))
        t.save(self.states_flipped_value, os.path.join(tensor_path, "states_flipped_value.pt"))

    @staticmethod
    def load(path: str, seq_len: int, max_samples: Optional[int] = None, load_processed=False) -> 'OthelloDataset':
        hf_dataset = load_from_disk(path)
        custom_dataset = OthelloDataset(None, seq_len, max_samples)
        custom_dataset._data = [{"tokens": tokens} for tokens in hf_dataset["tokens"][:max_samples]]
        
        tensor_path = f"{path}/tensors"
        
        custom_dataset.games_int = t.load(os.path.join(tensor_path, "games_int.pt"))
        custom_dataset.games_string = t.load(os.path.join(tensor_path, "games_string.pt"))
        
        if load_processed:
            # Load additional tensors
            custom_dataset.states = np.load(os.path.join(tensor_path, "states.npy"), allow_pickle=True)
            custom_dataset.valid_moves = t.load(os.path.join(tensor_path, "valid_moves.pt"))
            custom_dataset.flipped_states = np.load(os.path.join(tensor_path, "flipped_states.npy"), allow_pickle=True)
            custom_dataset.states_flipped_one_hot = t.load(os.path.join(tensor_path, "states_flipped_one_hot.pt"))
            custom_dataset.states_flipped_value = t.load(os.path.join(tensor_path, "states_flipped_value.pt"))
        
        return custom_dataset
