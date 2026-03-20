import os
import telebot
import torch
import soundfile as sf
from zipvoice.luxvoice import LuxTTS

# جلب التوكن من إعدادات GitHub Secrets
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
bot = telebot.TeleBot(TOKEN)

# تحميل النموذج على الـ CPU الخاص بسيرفرات GitHub
print("Loading LuxTTS Model...")
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=4)

user_data = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "أهلاً بك في بوت استنساخ الصوت! 🎙️\nأرسل لي مقطعاً صوتياً (3-5 ثوانٍ) أولاً.")

@bot.message_handler(content_types=['voice', 'audio'])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id if message.voice else message.audio.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    with open("prompt.ogg", 'wb') as new_file:
        new_file.write(downloaded_file)
    
    user_data[message.chat.id] = {'prompt': "prompt.ogg"}
    bot.reply_to(message, "✅ تم حفظ البصمة الصوتية! الآن أرسل النص الذي تريدني أن أنطقه بصوتك.")

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    chat_id = message.chat.id
    if chat_id in user_data:
        text = message.text
        bot.reply_to(message, "جاري استنساخ الصوت... انتظر ثوانٍ ⏳")
        
        try:
            encoded_prompt = lux_tts.encode_prompt(user_data[chat_id]['prompt'], duration=5, rms=0.01)
            final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)
            
            output_path = "output.wav"
            final_wav_np = final_wav.numpy().squeeze()
            sf.write(output_path, final_wav_np, 48000)
            
            with open(output_path, 'rb') as audio:
                bot.send_voice(chat_id, audio)
            
            del user_data[chat_id]
        except Exception as e:
            bot.reply_to(message, f"❌ حدث خطأ: {str(e)}")
    else:
        bot.reply_to(message, "⚠️ من فضلك أرسل مقطعاً صوتياً أولاً لاستخراج البصمة.")

bot.polling()
