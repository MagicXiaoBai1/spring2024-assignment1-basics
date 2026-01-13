from collections import defaultdict
import pickle
import os
from tqdm import tqdm
import logging

from cs336_basics.tokenizer.word_counts_generator.sub_step.MMapReaderStream import MMapReaderStream, FileReaderStream
from cs336_basics.tokenizer.word_counts_generator.sub_step.UTF8TextStream import UTF8TextStream
from cs336_basics.tokenizer.word_counts_generator.sub_step.Test2SampleStream import Test2SampleStream
from cs336_basics.tokenizer.word_counts_generator.sub_step.GPT2TokenStream import GPT2TokenStream

logger = logging.getLogger(__name__)

def generate_word_counts(file_path: str, output_path: str, special_tokens: list[str] = []):
    if len(special_tokens) > 0:
        endingToken = special_tokens[0]
    else:
        endingToken = "<|endoftext|>"
    # 构建流水线 🌊
    # reader = MMapReaderStream(file_path)
    reader = FileReaderStream(file_path)
    utf8_stream = UTF8TextStream(reader)
    test2_sample_stream = Test2SampleStream(utf8_stream, endingToken)
    token_stream = GPT2TokenStream(test2_sample_stream, special_tokens=special_tokens)
    
    # 估算总token数：根据经验，英文文本平均每个token约3-4个字节
    file_size = os.path.getsize(file_path)
    estimated_tokens = file_size // 3  # 保守估计
    
    word_count = {}
    shower =  tqdm(total=reader.total_bytes, desc="Processing tokens", unit="tokens")
    pre_bytes = 0
    for token in token_stream:
        shower.update(reader.now_bytes - pre_bytes)
        pre_bytes = reader.now_bytes

        word_count[token] = word_count.get(token, 0) + 1
    shower.close()
    
    # 保存结果
    with open(output_path, "wb") as f:
        pickle.dump(word_count, f)
    logger.info(f"Word counts saved to {output_path}")
    
    

if __name__ == "__main__":
    # generate_word_counts("data/TinyStoriesV2-GPT4-valid.txt", "output/word_counts_valid.pkl")
    # generate_word_counts("data/TinyStoriesV2-GPT4-train.txt", "output/word_counts_train.pkl")
    # generate_word_counts("data/owt_train.txt", "output/owt_train.txt_word_counts_train.pkl")
    

    
    """
    性能统计
        (base) love@loveSelf:~/python_code/spring2024-assignment1-basics$ /usr/bin/time -v /home/love/miniconda3/envs/dl_env/bin/python /home/love/python_code/spring2024-assignment1-basics/cs336_basics/tokenizer/main.py
        Processing word_count...
        Word counts saved to output/word_counts_train.pkl
                Command being timed: "/home/love/miniconda3/envs/dl_env/bin/python /home/love/python_code/spring2024-assignment1-basics/cs336_basics/tokenizer/main.py"
                User time (seconds): 241.68
                System time (seconds): 4.38
                Percent of CPU this job got: 99%
                Elapsed (wall clock) time (h:mm:ss or m:ss): 4:06.16
                Average shared text size (kbytes): 0
                Average unshared data size (kbytes): 0
                Average stack size (kbytes): 0
                Average total size (kbytes): 0
                Maximum resident set size (kbytes): 477280
                Average resident set size (kbytes): 0
                Major (requiring I/O) page faults: 0
                Minor (reclaiming a frame) page faults: 3022668
                Voluntary context switches: 1
                Involuntary context switches: 384
                Swaps: 0
                File system inputs: 0
                File system outputs: 1464
                Socket messages sent: 0
                Socket messages received: 0
                Signals delivered: 0
                Page size (bytes): 4096
                Exit status: 0
        (base) love@loveSelf:~/python_code/spring2024-assignment1-basics$ /usr/bin/time -v /home/love/miniconda3/envs/dl_env/bin/python /home/love/python_code/spring2024-assignment1-basics/cs336_basics/tokenizer/main.py
        Processing word_count...
        Word counts saved to output/owt_train.txt_word_counts_train.pkl
                Command being timed: "/home/love/miniconda3/envs/dl_env/bin/python /home/love/python_code/spring2024-assignment1-basics/cs336_basics/tokenizer/main.py"
                User time (seconds): 1397.28
                System time (seconds): 52.79
                Percent of CPU this job got: 99%
                Elapsed (wall clock) time (h:mm:ss or m:ss): 24:12.43
                Average shared text size (kbytes): 0
                Average unshared data size (kbytes): 0
                Average stack size (kbytes): 0
                Average total size (kbytes): 0
                Maximum resident set size (kbytes): 1364484
                Average resident set size (kbytes): 0
                Major (requiring I/O) page faults: 0
                Minor (reclaiming a frame) page faults: 26544611
                Voluntary context switches: 347
                Involuntary context switches: 33181
                Swaps: 0
                File system inputs: 23282264
                File system outputs: 189408
                Socket messages sent: 0
                Socket messages received: 0
                Signals delivered: 0
                Page size (bytes): 4096
                Exit status: 0

    """