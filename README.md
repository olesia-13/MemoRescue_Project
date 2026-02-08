#  MemoRescue

### AI-агент раннього виявлення дезорієнтації у людей з деменцією

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![AI](https://img.shields.io/badge/AI-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**MemoRescue** — це система комп’ютерного зору, що аналізує ходу людини та поведінкові патерни для виявлення потенційної дезорієнтації або небезпечного блукання.
Проєкт створено як демонстраційний AI-прототип для подальшого масштабування у системи міського відеоспостереження або медичного догляду.

---

##  Table of Contents

* [Problem](#-problem)
* [Solution](#-solution)
* [Why AI](#-why-ai)
* [Architecture](#-architecture)
* [Tech Stack](#-tech-stack)
* [Getting Started](#-getting-started)
* [Scaling Vision](#-scaling-vision)
* [Future Work](#-future-work)
* [Limitations](#-limitations)
* [Team](#-team)

---

##  Problem

Люди з когнітивними порушеннями можуть:

* втрачати орієнтацію
* блукати без мети
* потрапляти в небезпечні ситуації

Безперервний людський нагляд є складним та дорогим, тому потрібні автоматизовані рішення моніторингу.

---

##  Solution

MemoRescue:

1. Створює цифровий профіль ходи людини
2. Розпізнає її у відеопотоці
3. Аналізує траєкторію руху
4. Виявляє аномальну поведінку
5. Надсилає сповіщення через Telegram

---

##  Why AI

Система використовує:

* Pose Estimation (MediaPipe)
* Біомеханічні ознаки ходи
* Метрики схожості
* Аналіз часових послідовностей

Це дозволяє системі:

* адаптивно розпізнавати людей
* автоматично приймати рішення
* реагувати без ручного контролю

---

##  Architecture

### Registration Module

* введення даних користувача
* запис або завантаження відео
* витяг ключових точок
* формування gait signature
* збереження профілю

### Monitoring Module

* відеопотік з камери
* ідентифікація людини
* відстеження руху
* детекція аномалій:

  * zig-zag рух
  * тривала нерухомість
* Telegram alert

---

##  Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* Streamlit
* SciPy
* Telegram Bot API

---

##  Getting Started

###  Install dependencies

```bash
pip install -r requirements.txt
```


---

###  Configure Telegram

Створіть `.env`

```
TELEGRAM_BOT_TOKEN=your_token_here
```

---

###  Register user

```bash
streamlit run registration.py
```
Отримайте id вашого  Telegram чату через @userinfobot та впишіть його при реєстрації
Напишіть /start у боті [MemoRescue](http://t.me/MemoRescue_bot)

Профілі зберігаються у:

```
/database
```

---

###  Run monitoring

```bash
python monitor.py
```

Press `q` to exit.

---

##  Scaling Vision

Повноцінна система передбачає:

* інтеграцію з міськими камерами
* edge/server inference
* централізовані сповіщення
* масштабування на тисячі потоків

---

##  Future Work

* Deep learning gait recognition
* Multi-person tracking
* ML поведінкові класифікатори
* GPS інтеграція
* Mobile companion app
* Privacy-preserving inference

---

##  Limitations

* Однокамерний прототип
* Обмежена точність
* Не production-ready
* Спрощена модель поведінки

---

##  Team

Developed by A.I.D.O.

* Denis Tarasenko
* Nadia Lynovitsyka
* Olesia Osipova
* Vladyslava Polishchyk

---
