# SignSync AI 🤟
### AI That Connects Signs with Speech and Text

**SignSync AI** is an AI-powered communication and learning platform designed to reduce communication barriers between people who use sign language and those who do not understand it.

The project primarily focuses on **Indian Sign Language (ISL)** and aims to combine Computer Vision, Machine Learning, Natural Language Processing and Speech Technology to support sign recognition, text conversion, speech output and sign-language learning.

---

## 📌 Project Overview

Communication can become difficult when two people do not share a common language, especially when one person communicates using sign language.

SignSync AI aims to address this challenge through an intelligent system that can:

- Recognize sign gestures using a camera.
- Convert recognized signs into readable text.
- Convert text into speech.
- Provide a platform to learn and practice signs.
- Explore two-way communication between signs, text and speech.

## 🎯 Objectives

- Real-time recognition of selected Indian Sign Language gestures.
- Detection of static and selected dynamic signs.
- Conversion of recognized signs into text.
- Text-to-speech output.
- Sign-language learning and practice module.
- Performance feedback and progress tracking.
- Exploration of two-way sign and speech communication.

## 🏗️ System Architecture

### Sign-to-Text / Speech Pipeline

```text
       Camera Input
            |
            v
    Hand / Pose Detection
            |
            v
     Feature Extraction
            |
            v
   Sign Recognition Model
            |
            v
      Word / Sentence
            |
            v
        Text Output
            |
            v
       Speech Output
```

### Reverse Communication Pipeline

```text
      Speech / Text
            |
            v
     Text Processing
            |
            v
    Sign Representation
            |
            v
       Sign Display
```

## ✨ Key Features

| Feature | Description |
|---|---|
| Real-Time Sign Recognition | Recognizes selected sign gestures through camera input |
| Sign-to-Text | Converts recognized gestures into readable words |
| Text-to-Speech | Converts text output into spoken audio |
| Learning Module | Helps users learn and practice signs |
| Performance Feedback | Provides feedback or confidence information |
| Progress Tracking | Tracks learning performance |
| Two-Way Communication | Explores text/speech-to-sign communication |

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Machine Learning | CNN, LSTM, GRU, Transformer |
| NLP | Python NLP Libraries |
| Speech Processing | Speech-to-Text, Text-to-Speech |
| Backend | FastAPI |
| Database | PostgreSQL / MySQL |
| Frontend | React, HTML, CSS, JavaScript |
| Version Control | Git and GitHub |

*These technologies represent the proposed project stack; individual components may vary in the implemented version.*

## 👥 Team Modules

| Member | Responsibility |
|---|---|
| Member 1 | Computer Vision – Camera input, hand detection, landmarks and preprocessing |
| Member 2 | Machine Learning – Dataset preparation, model training and sign classification |
| Member 3 | NLP & Speech – Text processing, sentence formation, STT and TTS |
| Member 4 | AI Assistant
