

import pickle
from tqdm import tqdm

MergesUnit = tuple[int, int]

class BPETrainer:
    
    
    class MergesUnitChoiceFrequencyHeap:

        def __init__(self,a_is_bigger_than_b_merges_unit=None):
            self.a_is_bigger_than_b_merges_unit = a_is_bigger_than_b_merges_unit
            self.merges_unit_choise_frequency_heap: list[tuple[int, MergesUnit]] = []
            self.merges_unit2frequency_heap_index: dict[MergesUnit, int] = {}
        
        def insert_merges_unit(self, merges_unit: MergesUnit, frequency: int):
            self.merges_unit_choise_frequency_heap.append((frequency, merges_unit))
            self.merges_unit2frequency_heap_index[merges_unit] = len(self.merges_unit_choise_frequency_heap) - 1
            self.__heapify_up(len(self.merges_unit_choise_frequency_heap) - 1)
        
        
        
        def update_merges_unit(self, merges_unit: MergesUnit, frequency: int):
            
            if merges_unit not in self.merges_unit2frequency_heap_index:
                self.insert_merges_unit(merges_unit, frequency)
                return
            
            frequency_heap_index = self.merges_unit2frequency_heap_index[merges_unit]
            self.merges_unit_choise_frequency_heap[frequency_heap_index] = (frequency, merges_unit)
            self.__heapify_up(frequency_heap_index)
            self.__heapify_down(frequency_heap_index)
        
        
        def _delete_merges_unit(self, merges_unit: MergesUnit):
            del self.merges_unit2frequency_heap_index[merges_unit]
        
        def pop_max_frequency_merges_unit(self) -> tuple[MergesUnit, int]:
                        
            max_frequency, max_frequency_merges_unit = self.merges_unit_choise_frequency_heap[0]            
            del self.merges_unit2frequency_heap_index[max_frequency_merges_unit]
            self.merges_unit_choise_frequency_heap[0] = self.merges_unit_choise_frequency_heap[-1]
            self.merges_unit_choise_frequency_heap.pop()
            self.__heapify_down(0)
        
            return max_frequency_merges_unit, max_frequency
        
        def get_frequency(self, merges_unit: MergesUnit) -> int | None:
            if merges_unit not in self.merges_unit2frequency_heap_index:
                return 0
            frequency_heap_index = self.merges_unit2frequency_heap_index[merges_unit]
            return self.merges_unit_choise_frequency_heap[frequency_heap_index][0]
        
        def a_is_bigger_than_b (self, a: tuple[int, MergesUnit], b: tuple[int, MergesUnit]) -> bool:
            if a[0] > b[0]:
                return True
            elif a[0] < b[0]:
                return False
            return self.a_is_bigger_than_b_merges_unit(a[1], b[1])
        
        def __heapify_up(self, index: int):
            if index >= len(self.merges_unit_choise_frequency_heap):
                return
            if index == 0:
                return
            
            father_index = (index - 1) // 2
            if index > 0 and self.a_is_bigger_than_b(self.merges_unit_choise_frequency_heap[index], self.merges_unit_choise_frequency_heap[father_index]):
                # 交换
                self.merges_unit_choise_frequency_heap[index], self.merges_unit_choise_frequency_heap[father_index] = \
                    self.merges_unit_choise_frequency_heap[father_index], self.merges_unit_choise_frequency_heap[index]
                # 更新索引
                self.merges_unit2frequency_heap_index[self.merges_unit_choise_frequency_heap[index][1]] = index
                self.merges_unit2frequency_heap_index[self.merges_unit_choise_frequency_heap[father_index][1]] = father_index
                self.__heapify_up(father_index)
        
        
        def __heapify_down(self, index: int):
            if index >= len(self.merges_unit_choise_frequency_heap):
                return
            
            left_child_index = index * 2 + 1
            right_child_index = index * 2 + 2
            largest_index = index
            
            if left_child_index < len(self.merges_unit_choise_frequency_heap) and \
                self.a_is_bigger_than_b(self.merges_unit_choise_frequency_heap[left_child_index], self.merges_unit_choise_frequency_heap[largest_index]):
                largest_index = left_child_index
            if right_child_index < len(self.merges_unit_choise_frequency_heap) and \
                self.a_is_bigger_than_b(self.merges_unit_choise_frequency_heap[right_child_index], self.merges_unit_choise_frequency_heap[largest_index]):
                largest_index = right_child_index
            if largest_index != index:
                # 交换
                self.merges_unit_choise_frequency_heap[index], self.merges_unit_choise_frequency_heap[largest_index] = \
                    self.merges_unit_choise_frequency_heap[largest_index], self.merges_unit_choise_frequency_heap[index]
                # 更新索引
                self.merges_unit2frequency_heap_index[self.merges_unit_choise_frequency_heap[index][1]] = index
                self.merges_unit2frequency_heap_index[self.merges_unit_choise_frequency_heap[largest_index][1]] = largest_index
                self.__heapify_down(largest_index)

        
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.special_tokens: list[str] = special_tokens
        
        for idx, token in enumerate(special_tokens):
            self.vocab[idx] = token.encode('utf-8')
        
        for i in range(256):
            self.vocab[i + len(special_tokens)] = bytes([i])

        
        # 单次迭代用到的数据结构
        self.merges_unit2words: dict[MergesUnit, list[str]] = {} # 加速用的，不用完全准确的，只要不漏
        self.merges_unit_choise_frequency_heap: BPETrainer.MergesUnitChoiceFrequencyHeap
        self.bytes2token_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
    
    def train(self, word_count: dict[str, int], num_iterations: int = -1):
        for i in  self.special_tokens:
            word_count.pop(i, None)
        if num_iterations <= 0: num_iterations = self.vocab_size - len(self.vocab)
        
        self._cleanup_iteration_data()
        
        token_id_can_allocate = len(self.vocab)
        
        word2tokens: dict[str, list[int]] = {}
        for word, count in word_count.items():
            encoded_word = [self.bytes2token_id[bytes([b])] for b in word.encode('utf-8')]
            word2tokens[word] = encoded_word
            
        
        # Init merges_unit2count and merges_unit2words
        merges_unit2count: dict[MergesUnit, int] = {}
        for word, count in word_count.items():
            bytes_word = word.encode('utf-8')
            for i in range(len(bytes_word) - 1):
                pair = (self.bytes2token_id[bytes_word[i: i+1]], self.bytes2token_id[bytes_word[i+1: i+2]])
                
                # 下面是foreach pair的逻辑
                
                merges_unit2count[pair] = merges_unit2count.get(pair, 0) + count
                
                if pair not in self.merges_unit2words:
                    self.merges_unit2words[pair] = []
                self.merges_unit2words[pair].append(word)
        
        for k, v in merges_unit2count.items():
            self.merges_unit_choise_frequency_heap.insert_merges_unit(k, v)
        
        
        
        # Start iterations
        shower = tqdm(total=num_iterations, desc="now Processing merges", unit="times")
        update_interval = max(1, num_iterations // 100)
        for iteration in range(num_iterations):
            if iteration % update_interval == 0:
                shower.update(update_interval)  
                
            # Get the most frequent merges unit
            most_frequent_merges_unit, frequency = self.merges_unit_choise_frequency_heap.pop_max_frequency_merges_unit()
            
            merged_left = self.vocab[most_frequent_merges_unit[0]]
            merged_rigth = self.vocab[most_frequent_merges_unit[1]]
            
            self.merges.append((merged_left, merged_rigth))
            new_token_id = token_id_can_allocate
            token_id_can_allocate += 1
            self.vocab[new_token_id] = merged_left + merged_rigth
            
            
            
            affected_words = self.merges_unit2words.pop(most_frequent_merges_unit, [])
            for word in affected_words:
                # 更新 word 中的 most_frequent_merges_unit 出现的位置
                # 并更新相关的 merges_unit2count 和 merges_unit2words
                tokens_word = word2tokens[word]
                merged_start_indexs = []
                for i in range(len(tokens_word) - 1):
                    if (tokens_word[i], tokens_word[i+1]) == most_frequent_merges_unit:
                        # 找到一个需要合并的位置
                        if i > 0:
                            old_left_pair = (tokens_word[i-1], most_frequent_merges_unit[0])
                            new_old_left_pair_count = self.merges_unit_choise_frequency_heap.get_frequency(old_left_pair) - word_count[word]
                            self.merges_unit_choise_frequency_heap.update_merges_unit(old_left_pair, new_old_left_pair_count)
                            
                            left_pair = (tokens_word[i-1], new_token_id)
                            now_left_pair_count = self.merges_unit_choise_frequency_heap.get_frequency(left_pair) + word_count[word]
                            self.merges_unit_choise_frequency_heap.update_merges_unit(left_pair, now_left_pair_count)
                            self.merges_unit2words.setdefault(left_pair, []).append(word)
                            
                        if i < len(tokens_word) - 2:
                            old_right_pair = (most_frequent_merges_unit[1], tokens_word[i+2])
                            new_old_right_pair_count = self.merges_unit_choise_frequency_heap.get_frequency(old_right_pair) - word_count[word]
                            self.merges_unit_choise_frequency_heap.update_merges_unit(old_right_pair, new_old_right_pair_count)
                            
                            right_pair = (new_token_id, tokens_word[i+2])
                            now_right_pair_count = self.merges_unit_choise_frequency_heap.get_frequency(right_pair) + word_count[word]
                            self.merges_unit_choise_frequency_heap.update_merges_unit(right_pair, now_right_pair_count)
                            self.merges_unit2words.setdefault(right_pair, []).append(word)
                            
                            
                        # 更新 tokens_word
                        merged_start_indexs.append(i)
                        
                
                for merged_start_index in reversed(merged_start_indexs):
                    tokens_word[merged_start_index] = new_token_id
                    del tokens_word[merged_start_index+1]
        

        return self.vocab, self.merges


    def a_is_bigger_than_b_merges_unit (self, a:MergesUnit, b:MergesUnit) -> bool:
        if self.vocab[a[0]] > self.vocab[b[0]]:
            return True
        elif self.vocab[a[0]] < self.vocab[b[0]]:
            return False
        
        return self.vocab[a[1]] > self.vocab[b[1]]
    
    
    def _cleanup_iteration_data(self):
        self.merges_unit_choise_frequency_heap = BPETrainer.MergesUnitChoiceFrequencyHeap(self.a_is_bigger_than_b_merges_unit)
        self.merges_unit2words: dict[MergesUnit, list[str]] = {}
    
    


if __name__ == "__main__":
    
    with open("output/word_counts_train.pkl", "rb") as f:
        word_counts = pickle.load(f)
        
    BPETrainer(vocab_size=30522, special_tokens=["<|endoftext|>", "<|pad|>"]).train(word_counts, num_iterations=10000)