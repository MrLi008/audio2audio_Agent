我来为您提供一个完整的Gradio语音交互应用示例，实现语音输入转文本、调用OpenAI API获取回复，并自动播放语音回复。

```python
import gradio as gr
import openai
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import tempfile
import os
from typing import Iterator
import json

# 设置OpenAI API密钥
# 请将您的API密钥设置为环境变量或直接在这里设置
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"



# 初始化OpenAI客户端
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1/")


def transcribe_audio(audio_path: str) -> str:
    """将语音文件转为文本"""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"语音转文本错误: {e}")
        return f"语音识别错误: {str(e)}"

def text_to_speech_streaming(text: str) -> tuple:
    """将文本转为语音，返回音频文件和采样率"""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # 可以选择: alloy, echo, fable, onyx, nova, shimmer
            input=text,
            speed=1.0
        )
        
        # 将响应内容保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            for chunk in response.iter_bytes(chunk_size=4096):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        # 转换为Gradio兼容的格式
        audio = AudioSegment.from_file(tmp_file_path)
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        
        # 清理临时文件
        os.unlink(tmp_file_path)
        
        return sample_rate, samples.astype(np.float32) / (2**15)
        
    except Exception as e:
        print(f"文本转语音错误: {e}")
        return None, None

def generate_chat_response_streaming(user_input: str, history: list) -> Iterator[str]:
    """生成流式聊天响应"""
    messages = []
    
    # 添加上下文历史
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # 添加当前用户输入
    messages.append({"role": "user", "content": user_input})
    
    try:
        # 使用流式API
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 或使用 "gpt-4"
            messages=messages,
            stream=True,
            max_tokens=500,
            temperature=0.7
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield full_response
                
    except Exception as e:
        yield f"API调用错误: {str(e)}"

def process_audio_input(audio_path: str, history: list) -> tuple:
    """处理语音输入：转录 -> 生成回复 -> 转为语音"""
    # 1. 语音转文本
    user_text = transcribe_audio(audio_path)
    if not user_text or user_text.startswith("语音识别错误"):
        yield user_text, history, None, None
    
    # 2. 生成聊天响应（流式）
    full_response = ""
    for response_chunk in generate_chat_response_streaming(user_text, history):
        full_response = response_chunk
        yield user_text, history + [(user_text, full_response)], None, None
    
    # 3. 文本转语音
    sample_rate, audio_data = text_to_speech_streaming(full_response)
    
    # 4. 返回结果
    yield user_text, history + [(user_text, full_response)], sample_rate, audio_data

def process_text_input(user_text: str, history: list) -> tuple:
    """处理文本输入：生成回复 -> 转为语音"""
    # 1. 生成聊天响应（流式）
    full_response = ""
    for response_chunk in generate_chat_response_streaming(user_text, history):
        full_response = response_chunk
        yield user_text, history + [(user_text, full_response)], None, None
    
    # 2. 文本转语音
    sample_rate, audio_data = text_to_speech_streaming(full_response)
    
    # 3. 返回结果
    yield user_text, history + [(user_text, full_response)], sample_rate, audio_data

def clear_conversation():
    """清空对话历史"""
    return [], None, None, ""

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), title="智能语音助手") as demo:
    gr.Markdown("# 🎤 智能语音助手")
    gr.Markdown("### 支持语音输入，自动生成语音回复")
    
    # 状态变量
    chat_history = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="对话历史",
                height=400,
                bubble_full_width=False
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="文本输入",
                        placeholder="输入您的问题...",
                        lines=2
                    )
                with gr.Column(scale=1):
                    text_submit_btn = gr.Button("发送文本", variant="primary")
            
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="语音输入",
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#28a745",
                        waveform_progress_color="#155724",
                        skip_length=2,
                    )
                )
                audio_submit_btn = gr.Button("发送语音", variant="secondary")
            
            with gr.Row():
                clear_btn = gr.Button("清空对话", variant="stop")
                auto_play_toggle = gr.Checkbox(
                    label="自动播放语音回复",
                    value=True,
                    info="启用后会自动播放AI的语音回复"
                )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 语音回复")
            audio_output = gr.Audio(
                label="AI语音回复",
                type="numpy",
                autoplay=True,
                interactive=False
            )
            
            # 显示当前状态
            status_display = gr.Textbox(
                label="状态",
                value="等待输入...",
                interactive=False
            )
            
            # 设置说明
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("""
                1. **语音输入**: 点击录音按钮说话，然后点击"发送语音"
                2. **文本输入**: 在文本框中输入，点击"发送文本"
                3. **自动播放**: 勾选后会自动播放AI的语音回复
                4. **模型设置**: 默认使用 GPT-3.5-turbo + Whisper
                
                **功能特点**:
                - 🎙️ 实时语音转文本
                - 💬 流式文本回复
                - 🔊 文本转语音回复
                - 📱 响应式界面设计
                
                **注意**: 需要设置有效的OpenAI API密钥
                """)
    
    # 设置事件处理
    # 文本提交
    text_submit_btn.click(
        fn=process_text_input,
        inputs=[text_input, chat_history],
        outputs=[text_input, chat_history, audio_output, status_display],
        show_progress=True
    ).then(
        lambda: gr.update(value="等待输入..."),
        outputs=[status_display]
    )
    
    # 语音提交
    audio_submit_btn.click(
        fn=process_audio_input,
        inputs=[audio_input, chat_history],
        outputs=[text_input, chat_history, audio_output, status_display],
        show_progress=True
    ).then(
        lambda: gr.update(value="等待输入..."),
        outputs=[status_display]
    )
    
    # 清空对话
    clear_btn.click(
        fn=clear_conversation,
        outputs=[chat_history, audio_output, status_display, text_input]
    )
    
    # Enter键提交文本
    text_input.submit(
        fn=process_text_input,
        inputs=[text_input, chat_history],
        outputs=[text_input, chat_history, audio_output, status_display],
        show_progress=True
    ).then(
        lambda: gr.update(value="等待输入..."),
        outputs=[status_display]
    )
    
    # 状态更新函数
    def update_status(status):
        return status

# 启动应用
if __name__ == "__main__":
    # 检查API密钥
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 警告: 未设置OPENAI_API_KEY环境变量")
        print("请在运行前设置: export OPENAI_API_KEY='your-api-key'")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设置为True可生成公共链接
        debug=True
    )
```

## 安装依赖

创建一个 `requirements.txt` 文件：

```txt
gradio>=4.0.0
openai>=1.0.0
numpy>=1.21.0
soundfile>=0.11.0
pydub>=0.25.1
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 配置说明

1. **设置API密钥**：
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **可选配置**：
   - 修改 `model="gpt-3.5-turbo"` 为 `"gpt-4"` 使用更强大的模型
   - 修改 `voice="alloy"` 可以选择不同的语音：
     - `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

## 功能特点

1. **实时语音转文本**：使用OpenAI Whisper模型
2. **流式文本响应**：打字机效果显示回复
3. **自动语音合成**：将文本回复转为语音
4. **自动播放**：语音回复完成后自动播放
5. **双输入模式**：支持语音和文本两种输入方式
6. **对话历史**：保持完整的对话上下文

## 运行应用

```bash
python app.py
```

然后在浏览器中打开 `http://localhost:7860`

## 高级功能扩展

如果需要更高级的功能，可以考虑以下扩展：

1. **支持更多语音模型**：
   ```python
   # 可以使用其他TTS服务
   from gtts import gTTS  # Google TTS
   import edge_tts  # Microsoft Edge TTS
   ```

2. **实时语音流处理**：
   ```python
   # 使用WebSocket实现真正的实时双向语音
   ```

3. **多语言支持**：
   ```python
   # 指定语音识别和合成的语言
   transcript = client.audio.transcriptions.create(
       model="whisper-1",
       file=audio_file,
       language="zh"  # 中文
   )
   ```

这个应用提供了一个完整的语音交互解决方案，具有实时响应效果和良好的用户体验。
