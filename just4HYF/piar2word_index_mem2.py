
import mmap
import regex as re
from collections import defaultdict

# GPT-2 分词正则（与官方一致）
GPT2_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def utf8_byte_encode(text: str):
    """将字符串转为 GPT-2 风格的字节 token 列表 + </w>"""
    return tuple([f"b{b}" for b in text.encode("utf-8")] + ["</w>"])

def get_pairs(word):
    """返回相邻符号对集合"""
    return set(zip(word, word[1:]))

def is_chunk_complete_utf8(text: bytes):
    """检查文本块是否为完整的 UTF-8 编码"""
    try:
        text.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
complete_end = end
while offset < complete_end and not is_chunk_complete_utf8(mm[offset:end]):
    complete_end -= 1

chunk_text = mm[offset:complete_end].decode("utf-8", errors="ignore")
def train_bpe_from_large_file(filepath, min_freq=2):
    """
    直接从超大文本文件训练 BPE 初始 pair 频次（不存 tokens）
    返回: pair_freqs: dict[(str,str), int]
    """
    pair_freqs = defaultdict(int)
    buffer = ""  # 用于拼接跨块的残片

    with open(filepath, "rb") as f:  # 注意：mmap 需要二进制模式
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            offset = 0
            chunk_size = 64 * 1024 * 1024  # 64MB chunks

            while offset < file_size:
                # 读取一个 chunk（bytes）
                end = min(offset + chunk_size, file_size)
                chunk_bytes = mm[offset:end]

                # 尝试完整解码 UTF-8；若失败，回退到最后一个完整字符
                length = len(chunk_bytes)
                complete_end = end
                while offset < complete_end and not is_chunk_complete_utf8(mm[offset:end]):
                    complete_end -= 1

                chunk_text = mm[offset:complete_end].decode("utf-8", errors="ignore")
                # 拼接上一轮残片
                text = buffer + chunk_text

                # 如果不是最后一块，保留末尾可能不完整的 token
                if offset < file_size:
                    # 找到最后一个空白符（安全断点）
                    safe_pos = -1
                    for i in range(len(text) - 1, -1, -1):
                        if text[i] in (' ', '\n', '\t', '\r'):
                            safe_pos = i
                            break
                    if safe_pos != -1:
                        process_text = text[:safe_pos + 1]
                        buffer = text[safe_pos + 1:]
                    else:
                        # 没有安全断点？暂时全处理（风险低，因 chunk 大）
                        process_text = text
                        buffer = ""
                else:
                    # 最后一块，处理全部
                    process_text = text
                    buffer = ""

                # 🔥 核心：流式正则匹配 + BPE 初始化
                for match in re.finditer(GPT2_REGEX, process_text):
                    token = match.group()
                    if not token.strip():
                        continue  # 跳过纯空白（可选）

                    # 转为 BPE 初始符号序列（字节级）
                    symbols = utf8_byte_encode(token)
                    # 统计所有相邻 pair
                    for pair in get_pairs(symbols):
                        pair_freqs[pair] += 1

            # 处理最后残片
            if buffer:
                for match in re.finditer(GPT2_REGEX, buffer):
                    token = match.group()
                    if not token.strip():
                        continue
                    symbols = utf8_byte_encode(token)
                    for pair in get_pairs(symbols):
                        pair_freqs[pair] += 1

    # 过滤低频 pair（可选）
    if min_freq > 1:
        pair_freqs = {k: v for k, v in pair_freqs.items() if v >= min_freq}

    return pair_freqs



