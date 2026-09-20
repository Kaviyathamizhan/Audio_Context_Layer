"""Sound classes, scene definitions and the small causal/alert knowledge table.

All foreground classes are ESC-50 categories. Scenes group classes into plausible
environments. The knowledge table is the *label source* for causal and alert
questions: it is hand-written (not human-verified per sample) and is documented
as a limitation in the report.
"""

SR = 16000            # sample rate used everywhere
DUR = 10.0            # every scene is exactly 10 s (fits the AST input window)
N_SAMPLES = int(SR * DUR)
STEP = 0.1            # temporal resolution of the frame-level detector (s)
T_STEPS = int(round(DUR / STEP))   # 100

# esc50 category -> human-readable label used in questions
FG_LABELS = {
    # street
    "car_horn": "car horn", "siren": "siren", "engine": "engine",
    "footsteps": "footsteps", "train": "train", "helicopter": "helicopter",
    # home
    "door_wood_knock": "door knock", "clock_alarm": "alarm clock",
    "glass_breaking": "glass breaking", "vacuum_cleaner": "vacuum cleaner",
    "sneezing": "sneeze", "coughing": "cough",
    # farm / countryside
    "rooster": "rooster", "crow": "crow", "cow": "cow", "sheep": "sheep",
    "dog": "dog bark", "chirping_birds": "birds chirping",
    # celebration / public event
    "fireworks": "fireworks", "clapping": "clapping", "laughing": "laughter",
    "church_bells": "church bells",
}
CLASSES = list(FG_LABELS)                       # fixed order = output index
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
LABEL2CLASS = {v: k for k, v in FG_LABELS.items()}
N_CLASSES = len(CLASSES)

SCENES = {
    "street": {"label": "a busy street", "bg": "wind",
               "fg": ["car_horn", "siren", "engine", "footsteps", "train", "helicopter"]},
    "home": {"label": "inside a home", "bg": "clock_tick",
             "fg": ["door_wood_knock", "clock_alarm", "glass_breaking", "vacuum_cleaner", "sneezing", "coughing"]},
    "farm": {"label": "a farm or countryside", "bg": "rain",
             "fg": ["rooster", "crow", "cow", "sheep", "dog", "chirping_birds"]},
    "celebration": {"label": "a celebration or public event", "bg": "crickets",
                    "fg": ["fireworks", "clapping", "laughing", "church_bells"]},
}
SCENE_NAMES = list(SCENES)
SCENE_LABELS = {k: v["label"] for k, v in SCENES.items()}
SCENELABEL2NAME = {v: k for k, v in SCENE_LABELS.items()}
CLASS2SCENE = {c: s for s, d in SCENES.items() for c in d["fg"]}
BG_CLASSES = sorted({d["bg"] for d in SCENES.values()})

# (cause text, is_alert_worthy_for_a_deaf_listener)
KNOWLEDGE = {
    "car_horn": ("a driver is warning or signalling other road users", True),
    "siren": ("an emergency vehicle is passing or approaching", True),
    "engine": ("a motor vehicle or engine is running nearby", False),
    "footsteps": ("someone is walking nearby", False),
    "train": ("a train is passing by", False),
    "helicopter": ("a helicopter is flying overhead", False),
    "door_wood_knock": ("a visitor is trying to get attention at the door", True),
    "clock_alarm": ("an alarm was set to wake someone or mark a time", True),
    "glass_breaking": ("a glass object was dropped or smashed", True),
    "vacuum_cleaner": ("someone is cleaning the floor", False),
    "sneezing": ("a person's nose is irritated, for example by a cold or allergy", False),
    "coughing": ("a person is clearing their throat or is unwell", False),
    "rooster": ("a rooster is announcing the morning or marking its territory", False),
    "crow": ("a crow is calling to other birds", False),
    "cow": ("a cow is calling out on a farm", False),
    "sheep": ("a sheep is calling to its flock", False),
    "dog": ("a dog is reacting to something nearby", False),
    "chirping_birds": ("birds are singing or communicating nearby", False),
    "fireworks": ("people are celebrating a festival or event", False),
    "clapping": ("an audience is applauding", False),
    "laughing": ("someone found something amusing", False),
    "church_bells": ("a church is marking the hour or a ceremony", False),
}
CAUSE = {c: v[0] for c, v in KNOWLEDGE.items()}
CAUSE2CLASS = {v: k for k, v in CAUSE.items()}
ALERT = {c for c, v in KNOWLEDGE.items() if v[1]}
assert len(CAUSE2CLASS) == len(CAUSE), "causes must be unique"
assert set(KNOWLEDGE) == set(CLASSES)
assert set(CLASS2SCENE) == set(CLASSES)
