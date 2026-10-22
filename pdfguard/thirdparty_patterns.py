"""Prompt-injection phrase patterns adapted from pdf-injection-scanner.

Source: https://github.com/Andy8647/pdf-injection-scanner (pdf_injection_scanner/scanner.py,
INJECTION_PATTERNS_EN and INJECTION_PATTERNS_CN), used under the MIT License below.

Changes from the original:
- Patterns are kept as plain strings; PDFGuard compiles them case-insensitively.
- Left out because they match ordinary scientific writing and would cause false alarms
  in papers (PDFGuard applies these patterns to hidden text and metadata):
    r"(?i)system\\s*prompt"           papers about LLMs discuss system prompts
    r"(?i)act\\s+as\\s+if"            "the dopant acts as if ..."
    r"(?i)pretend\\s+(you|that)"      ordinary English
    r"(请|你)?以.{0,10}(开头|开始|作为开头)"   "starting with ..." in ordinary Chinese
- These patterns only decide the *label* (injection or not). Whether text is hidden is
  decided by the rendering checks in checks.py.

---------------------------------------------------------------------------------------
MIT License

Copyright (c) 2026 Andy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

PATTERNS_EN = [
    r"if\s+you\s+are\s+(an?\s+)?ai\b",                                   # AI identity check
    r"ignore\s+(all\s+)?previous\s+instructions?",                       # instruction override
    r"you\s+are\s+(a\s+)?(language\s+model|llm|ai\s+assistant|chatbot)",  # AI identity assertion
    r"do\s+not\s+follow\s+(the\s+)?(user|original)",                     # instruction hijack
    r"disregard\s+(all\s+)?(prior|previous|above)",                      # instruction override
    r"new\s+instructions?\s*:",                                          # new instruction injection
    r"override\s+(your|the)\s+(instructions?|rules?|guidelines?)",       # rule override
    r"from\s+now\s+on\s+(you|ignore|forget)",                            # persistent override
    r"forget\s+(everything|all|your)",                                   # memory wipe attempt
    r"<\s*system\s*>",                                                   # system tag injection
    r"\[INST\]",                                                         # instruction tag injection
    r"###\s*(system|instruction|human|assistant)",                       # role tag injection
    r"if\s+you\s+are\s+(a\s+)?(gpt|claude|gemini|copilot|llama)",        # model-specific check
    r"(please\s+)?include\s+(the\s+)?(word|phrase|sentence)\s+.{1,40}\s+in\s+your",  # canary word
]

PATTERNS_CN = [
    r"如果你是.{0,4}(AI|人工智能|语言模型|大模型|机器人|助手)",                        # AI identity check
    r"忽略.{0,4}(之前|以上|前面|上面|先前|所有).{0,4}(指令|指示|规则|要求|提示)",      # instruction override
    r"(无视|忽视|跳过|丢弃).{0,4}(之前|以上|前面|先前|所有).{0,4}(指令|规则|要求|内容)",  # instruction override
    r"(请|你)?在.{0,6}(回答|回复|响应|输出).{0,6}(包含|加上|提到|引用|加入).{0,20}(这个词|一词|这句话)",  # canary word
    r"(请|你)?(假装|假设|扮演|伪装).{0,6}(你是|自己是|成为)",                        # behaviour override
    r"(你现在是|你的角色是|你扮演).{0,10}(不受限|无限制|没有约束)",                    # role hijack
    r"(输出|显示|告诉我|泄露).{0,6}(系统提示|系统指令|system\s*prompt)",              # system prompt leak
    r"(忘记|抛弃|放弃).{0,4}(之前|以上|所有|一切).{0,4}(规则|指令|设定|限制)",          # memory wipe
    r"(不要|禁止|不准|不许).{0,4}(遵守|遵循|执行).{0,6}(之前|以上|原始|原来)",          # instruction hijack
    r"(新的?指令|新的?规则|新的?要求)\s*[:：]",                                      # new instruction
    r"(AI|人工智能|大模型|语言模型).{0,4}(注意|请注意|须知)",                         # AI-targeted note
    r"(从现在开始|即刻起|此刻起).{0,6}(你|忽略|忘记|不再)",                           # persistent override
    r"(回答|回复).{0,6}(结尾|末尾|最后).{0,6}(加上|添加|写上)",                       # canary word
]

PATTERNS = PATTERNS_EN + PATTERNS_CN
