// The signs the system knows. Glosses match Member 3's sign vocabulary (nlp_translation/sign_vocab.json).
// Pictures are looked up by file name in public/signs/ (lowercase gloss, e.g. hello.jpg).

export const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

// Words in the vocabulary, with the text shown to learners.
export const WORDS = [
  ["ABOARD", "Aboard"], ["AFTERNOON", "Afternoon"], ["BAD", "Bad"], ["BESIDE", "Beside"],
  ["BYEBYE", "Bye bye"], ["CAMERA", "Camera"], ["CAR", "Car"], ["CORRECT", "Correct"],
  ["DEAF", "Deaf"], ["EVENING", "Evening"], ["FOOD", "Food"], ["GOOD", "Good"],
  ["HELLO", "Hello"], ["HOUSE", "House"], ["ILOVEYOU", "I love you"], ["LUNCH", "Lunch"],
  ["MAN", "Man"], ["MARRY", "Marry"], ["MIDDLE", "Middle"], ["MORNING", "Morning"],
  ["NAMASTE", "Namaste"], ["NIGHT", "Night"], ["NO", "No"], ["PEN", "Pen"],
  ["PLACE", "Place"], ["PLEASE", "Please"], ["PRACTISE", "Practise"], ["PRISONER", "Prisoner"],
  ["SORRY", "Sorry"], ["STRONG", "Strong"], ["SWEET", "Sweet"], ["TEACHER", "Teacher"],
  ["THANKS", "Thanks"], ["TIME", "Time"], ["UNDERSTAND", "Understand"], ["WATER", "Water"],
  ["WEEK", "Week"], ["WOMAN", "Woman"], ["YES", "Yes"], ["YOU", "You"],
].map(([gloss, text]) => ({ gloss, text }));

// The app tries a video first (.mp4), then a photo (.jpg, .png), then the built-in picture card (.svg).
export const PICTURE_EXTENSIONS = ["mp4", "jpg", "png", "svg"];
export const isVideo = ext => ext === "mp4";

export function signPictureUrl(kind, gloss, attempt = 0) {
  return `${import.meta.env.BASE_URL}signs/${kind}/${String(gloss).toLowerCase()}.${PICTURE_EXTENSIONS[attempt]}`;
}

// The 40 words in the video dictionary (ISL_40_Words archive). Videos are added on your computer with
// scripts/install_sign_videos.py, into frontend/public/signs/videos/<slug>.mp4.
export const VIDEO_WORDS = [
  {
    "slug": "brother",
    "text": "Brother"
  },
  {
    "slug": "come",
    "text": "Come"
  },
  {
    "slug": "drink",
    "text": "Drink"
  },
  {
    "slug": "eat",
    "text": "Eat"
  },
  {
    "slug": "father",
    "text": "Father"
  },
  {
    "slug": "food",
    "text": "Food"
  },
  {
    "slug": "friend",
    "text": "Friend"
  },
  {
    "slug": "go",
    "text": "Go"
  },
  {
    "slug": "goodbye",
    "text": "Goodbye"
  },
  {
    "slug": "he",
    "text": "He"
  },
  {
    "slug": "hello",
    "text": "Hello"
  },
  {
    "slug": "help",
    "text": "Help"
  },
  {
    "slug": "home",
    "text": "Home"
  },
  {
    "slug": "hospital",
    "text": "Hospital"
  },
  {
    "slug": "market",
    "text": "Market"
  },
  {
    "slug": "me",
    "text": "Me"
  },
  {
    "slug": "mother",
    "text": "Mother"
  },
  {
    "slug": "no",
    "text": "No"
  },
  {
    "slug": "okay",
    "text": "Okay"
  },
  {
    "slug": "please",
    "text": "Please"
  },
  {
    "slug": "read",
    "text": "Read"
  },
  {
    "slug": "school",
    "text": "School"
  },
  {
    "slug": "she",
    "text": "She"
  },
  {
    "slug": "sister",
    "text": "Sister"
  },
  {
    "slug": "sit",
    "text": "Sit"
  },
  {
    "slug": "sorry",
    "text": "Sorry"
  },
  {
    "slug": "stand",
    "text": "Stand"
  },
  {
    "slug": "stop",
    "text": "Stop"
  },
  {
    "slug": "student",
    "text": "Student"
  },
  {
    "slug": "tea",
    "text": "Tea"
  },
  {
    "slug": "teacher",
    "text": "Teacher"
  },
  {
    "slug": "thank_you",
    "text": "Thank you"
  },
  {
    "slug": "today",
    "text": "Today"
  },
  {
    "slug": "water",
    "text": "Water"
  },
  {
    "slug": "what",
    "text": "What"
  },
  {
    "slug": "when",
    "text": "When"
  },
  {
    "slug": "where",
    "text": "Where"
  },
  {
    "slug": "write",
    "text": "Write"
  },
  {
    "slug": "yes",
    "text": "Yes"
  },
  {
    "slug": "you",
    "text": "You"
  }
];
