Project PRD: "Remi" - Local Gemini Assistant with GUI
1. 项目概述 (Overview)
构建一个基于 Windows 本地运行 的桌面级 AI 助手应用。
该应用拥有现代化的图形界面 (GUI)，允许用户自定义 API Key 和模型，支持自定义 AI 人设（System Persona）。核心差异化功能是利用 GibsonAI/Memori 实现跨会话的长期记忆，并使用 LangChain 进行业务编排。
2. 技术栈 (Tech Stack)
前端 UI: Streamlit (使用 st.chat_message, st.sidebar 等组件构建现代化聊天界面)
核心逻辑: Python 3.14
LLM 编排: langchain, langchain-google-genai
核心模型: Google Gemini  (动态配置，默认为 gemini-2.5-flash)
记忆引擎: memori (位于项目根目录下的本地memori-main文件夹)
数据存储:
  长期记忆: SQLite (local_memory.db 由 Memori 管理)
  会话展示: st.session_state (Streamlit 运行时状态)
3. 用户体验设计 (UI/UX)
3.1 侧边栏 (Settings Sidebar)
用户进入应用后，左侧为配置区：
API 配置：
输入框: Google API Key (密码掩码显示 type="password")。
输入框: Model Name (默认填入 gemini-2.5-flash，允许用户修改)。
人设定义 (Persona)：
多行文本框: "AI 人设/系统提示词"。
默认值: "你是一个女性私人助理，你的名字叫 Remi。你拥有对用户的长期记忆，能生成个性化的回答。"
记忆管理：
显示当前记忆数据库的大小或路径。
按钮: "清除当前会话显示" (Reset Chat)。
3.2 主聊天区 (Main Chat Interface)
对话流：经典的 IM 布局。用户在右侧（绿色气泡），AI 在左侧（灰色/彩色气泡）。最近一个消息可以修改重新发送。
输入框：底部固定聊天输入栏。
状态反馈：AI 生成时显示 "Thinking & Remembering..." 加载动画。
4. 核心功能逻辑 (Functional Requirements)
4.1 初始化与配置
应用启动时，检查 st.session_state 中是否有 API Key。如果没有，提示用户在侧边栏输入。
Memori 初始化：
必须引用本地 memori 文件夹。
初始化 Memori 实例，连接到 sqlite:///local_memory.db。
关键点：确保 Memori 的内部处理（Memory Processing）也使用用户输入的 Gemini Key，而不是默认去寻找 OpenAI Key。
4.2 对话处理流程 (RAG Loop)
当用户发送消息 user_input 时：
Retrieve (回忆)：
调用 memori.retrieve(user_input) 获取相关历史记忆 memory_context。
Augment (增强)：
构建 Prompt。
System Message = 侧边栏的“人设定义” + " \n\n【长期记忆参考】:\n" + memory_context。
Human Message = user_input。
Generate (生成)：
使用 ChatGoogleGenerativeAI (LangChain) 调用 Gemini API。
支持流式输出 (Streaming) 到前端界面（可选，若太复杂可先做非流式）。
Store (存储)：
AI 回复完成后，调用 memori.add_interaction(user_input, ai_response)。
这一步必须在后台执行，不应阻塞用户阅读当前回复。
5. 目录结构规范
code
Text
project_root/
├── app.py              # Streamlit 主程序入口 (包含 UI 和 逻辑)
├── requirements.txt    # 依赖: streamlit, langchain-google-genai, litellm
└── memori/             # (现有的本地 Memori SDK 文件夹)
6. 给 Cursor 的特别指令 (Instructions for Cursor)
Import 处理：请注意 memori 文件夹在根目录。使用 sys.path.append 或直接 from memori import Memori 确保能正确导入。
Streamlit 状态管理：
使用 st.session_state.messages 来存储当前显示的聊天记录（用于 UI 渲染）。
不要把 Memori 实例反复重新初始化。使用 @st.cache_resource 装饰器来缓存 Memori 的连接实例，确保数据库连接稳定。
API Key 安全：不要把 API Key 硬编码。必须从 st.sidebar 的输入框获取，并传递给 LangChain 和 Memori。
错误处理：如果用户未输入 Key 就发送消息，弹出 st.warning("请先在左侧填写 API Key")。
LangChain 集成：使用 ChatGoogleGenerativeAI 类。请确保 temperature 可配置或设为 0.7。