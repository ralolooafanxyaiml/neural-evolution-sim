# --- database.py ---

# 1. BASIS
ARCHETYPES = {
    "BIG_CAT": [1, 1, 1, 3, 1],
    "CANINE": [1, 1, 1, 2, 1],
    "HERBIVORE_MEGA": [1, 4, 1, 4, 2],
    "GRAZER": [1, 1, 1, 3, 2],
    "RODENT": [1, 1, 1, 1, 3],
    "RAPTOR_BIRD": [1, 3, 3, 2, 1],
    "WATER_BIRD": [1, 3, 2, 2, 3],
    "SNAKE_LIKE": [0, 2, 1, 2, 1],
    "LIZARD_SMALL": [0, 2, 1, 1, 3],
    "AMPHIBIAN_STD": [0, 4, 2, 1, 3],
    "FISH_PREDATOR": [0, 2, 2, 3, 1],
    "FISH_PREY": [0, 2, 2, 1, 2],
    "INSECT_GIANT": [0, 2, 1, 1, 3],
    "WHALE_LIKE": [1, 4, 2, 4, 1],
    "PRIMATE": [1, 1, 1, 2, 3],
    "BEAR_LIKE": [1, 1, 1, 3, 3],
    "MARINE_REPTILE": [0, 2, 2, 4, 1],
    "MARSUPIAL": [1, 1, 1, 2, 2],
    "DEEP_SEA_FISH": [0, 4, 2, 1, 1],
    "WINGED_INSECT": [0, 2, 3, 0, 3]
}

# 2. EVOLUTION MAPPING
EVOLUTION_MAPPING = {
    0: [
        "Significant thickening of the dermal layer and fur density for extreme cold insulation.",
        "Development of a dense undercoat and increased subcutaneous fat reserves.",
        "Adaptation to cold by minimizing surface area and maximizing thermal retention."
    ],
    1: [
        "Skin hardens into keratinous scales or a protective shell against heat/toxins.",
        "Rapid growth of tough scales or a carapace to prevent dehydration.",
        "Cellular structure becomes highly resistant to corrosive elements."
    ],
    2: [
        "Shift to a prolonged torpor or hibernation state to conserve energy.",
        "Reduction in basal metabolic rate by 40% to survive scarcity.",
        "Evolution of a super-efficient energy storage organ."
    ],
    3: [
        "Lungs/gills become hyper-efficient, maximizing oxygen uptake.",
        "Development of larger wings or stronger aquatic fins for rapid migration.",
        "Evolution of a secondary, low-oxygen tolerance organ."
    ],
    4: [
        "Glands develop to secrete defensive toxins with warning coloration.",
        "Cellular mechanisms rapidly evolve to neutralize pollutants.",
        "Development of a bitter taste or unpalatable texture."
    ],
    5: [
        "Significant enlargement of eyes or auditory organs.",
        "Development of specialized sensory organs (e.g., electroreception).",
        "Increased brain capacity for processing complex sensory data."
    ]
}

ATTRIBUTE_CATEGORIES = {
    0: "SKIN_INSULATION", 
    1: "SKIN_ARMOR", 
    2: "METABOLISM",
    3: "RESPIRATORY",
    4: "DEFENSE",
    5: "SENSORY"
}

# 3. DATABASES
ANIMAL_DATABASE = {}
ANIMAL_DATABASE["dragon"] = [0, 2, 3, 4, 1]
ANIMAL_DATABASE["alien"] = [1, 4, 1, 2, 3]

THREAT_DATABASE = {}

def add_animals(archetype_key, animal_list):
    for animal in animal_list:
        if archetype_key in ARCHETYPES:
            ANIMAL_DATABASE[animal] = ARCHETYPES[archetype_key]

def add_threats(threat_id, keywords):
    for word in keywords:
        THREAT_DATABASE[word] = threat_id

# LISTS FOR DATABASES
add_animals("BIG_CAT", ["lion", "tiger", "leopard", "jaguar", "cheetah", "panther", "cougar", "lynx"])
add_animals("CANINE", ["wolf", "dog", "fox", "coyote", "jackal", "hyena", "dingo"])
add_animals("HERBIVORE_MEGA", ["elephant", "rhino", "hippo", "giraffe", "dinosaur", "brachiosaurus", "mammoth"])
add_animals("GRAZER", ["horse", "cow", "zebra", "deer", "moose", "camel", "buffalo", "gazelle", "donkey", "sheep", "goat"])            
add_animals("RODENT", ["mouse", "rat", "hamster", "squirrel", "beaver", "rabbit", "hare", "guinea pig"])
add_animals("RAPTOR_BIRD", ["eagle", "hawk", "falcon", "owl", "vulture", "condor"])
add_animals("WATER_BIRD", ["penguin", "duck", "swan", "goose", "pelican", "seagull", "albatross"])
add_animals("SNAKE_LIKE", ["snake", "cobra", "python", "viper", "anaconda", "boa"])
add_animals("LIZARD_SMALL", ["lizard", "gecko", "chameleon", "iguana", "skink"])
add_animals("AMPHIBIAN_STD", ["frog", "toad", "salamander", "newt", "axolotl"])
add_animals("FISH_PREDATOR", ["shark", "great white", "barracuda", "swordfish", "piranha"])
add_animals("FISH_PREY", ["goldfish", "salmon", "tuna", "trout", "cod", "sardine", "clownfish"])
add_animals("WHALE_LIKE", ["whale", "blue whale", "orca", "dolphin", "beluga", "manatee"])
add_animals("PRIMATE", ["human", "monkey", "chimpanzee", "gorilla", "orangutan", "lemur", "baboon"])
add_animals("BEAR_LIKE", ["bear", "grizzly", "polar bear", "panda", "koala"])
add_animals("MARINE_REPTILE", ["crocodile", "alligator", "caiman", "sea turtle", "ichthyosaur"])
add_animals("MARSUPIAL", ["kangaroo", "wallaby", "opossum", "tasmanian devil"])
add_animals("DEEP_SEA_FISH", ["anglerfish", "lanternfish", "hagfish", "jellyfish"])
add_animals("WINGED_INSECT", ["fly", "bee", "wasp", "moth", "butterfly", "dragonfly"])

add_threats(1, ["cold", "freezing", "ice", "ice age", "snow", "blizzard", "arctic", "winter", "chill"])
add_threats(2, ["heat", "hot", "fire", "lava", "magma", "volcano", "sun", "global warming", "desert", "dry"])
add_threats(3, ["toxin", "toxic", "poison", "pollution", "plastic", "radiation", "nuclear", "virus", "bacteria"])
add_threats(4, ["scarcity", "famine", "starvation", "hunger", "no food", "empty", "poverty"])

add_threats(5, ["air", "no air", "oxygen", "suffocation", "space", "underwater", "predator", "hunter", "attack"])

