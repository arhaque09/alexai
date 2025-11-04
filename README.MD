# Alex AI — Personality-Based Terminal Chat

**Alex AI** is a terminal-based conversational program that dynamically changes **text colors** to match the AI’s current **personality tone**. Each interaction feels unique, as Alex can shift between **loving**, **harsh**, or **direct** tones—each with its own visual and emotional flair.

---

## 🌈 Features

* **Dynamic Personality Colors:** The terminal color theme changes as Alex’s tone changes.
* **Flexible Startup Options:**

  * Start with a **random personality**
  * Use a **predefined personality**
  * Create your **own custom personality**
* **Interactive Conversation:** Chat directly with Alex through the terminal.
* **Trainable Personality Model:** Improve or modify Alex’s responses by retraining the model using your own data.

---

## 🧠 Training Alex AI

You can train or fine-tune Alex’s responses by editing the `rawdata.py` file and then running the following commands in order:

# Commands to run:

```bash
python src/prepare_data.py
python src/train_qlora.py
python src/infer.py --base_model Qwen/Qwen2.5-3B-Instruct --adapter_dir outputs\alex-qlora --tone direct --user "Give me 3 resume tips"
python src/runtime/main.py # runs the actual file
```

---

## 💬 Personalities

Each personality defines how Alex interacts:

* **Random** — Unpredictable and spontaneous (*text color: white*)
* **Loving** — Gentle and encouraging (*text color: pastel pink*)
* **Neutral** — Balanced and objective (*text color: gray*)
* **Warm** — Kind and comforting (*text color: light orange*)
* **Playful** — Energetic and teasing (*text color: cyan*)
* **Formal** — Polite and professional (*text color: dark gray*)
* **Casual** — Relaxed and friendly (*text color: green*)
* **Authoritative** — Commanding and confident (*text color: blue*)
* **Cautious** — Careful and thoughtful (*text color: amber*)
* **Direct** — Straightforward and concise (*text color: yellow*)
* **Diplomatic** — Polite and tactful (*text color: light green*)
* **Curious** — Inquisitive and engaged (*text color: sky blue*)
* **Witty/Dry** — Clever and subtly humorous (*text color: gray*)
* **Reflective** — Thoughtful and introspective (*text color: navy blue*)
* **Harsh** — Blunt and strict (*text color: red*)
* **Create new Personality** — Custom tone defined by the user (*text color: customizable*)


---

## ⚙️ Requirements

* Python 3.10+
* PyTorch
* Transformers
* Accelerate
* Datasets
* QLoRA-compatible environment

---

## 🧡 Credits

Developed with passion by **Arisha Haque**, blending emotional intelligence and AI learning to create an adaptive, color-changing terminal companion.
