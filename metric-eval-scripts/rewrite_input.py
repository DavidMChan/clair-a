import csv
import random
import sys

rephrases = [
    "Just because",
    "No specific reason",
    "No reason at all",
    "It just is",
    "For no reason",
    "Not for any special reason",
    "No particular motive",
    "No real reason",
    "No thought behind it",
    "No noteworthy reason",
    "Just on a whim",
    "Without any particular cause",
    "Not really a reason, just felt like it",
    "No thought behind it whatsoever",
    "No strong rationale, just went with it",
    "Just an impulse",
    "No real reason, just something that came to mind",
    "I didn’t have a particular reason, it just seemed like the thing to do",
    "Honestly, I can’t think of any specific reason why, it just happened like that",
    "I couldn’t tell you why exactly, there wasn’t a real reason for it",
    "There wasn’t any thought behind it, I just did it without thinking too much about it",
    "I wasn’t driven by anything in particular, I just decided to go with that",
    "There was no plan, no particular thought, just a decision I made on a whim",
    "Honestly, there was no specific logic or rationale. It just felt like something I should do at the moment",
    "To be truthful, I didn’t have any solid reasoning for it. It just kind of came to me, and I went with it",
    "If you ask me why, I couldn’t give you a definitive answer. It just happened, no reason behind it",
    "I didn’t spend much time thinking about it. No reason crossed my mind, I just acted in the moment",
    "There was no purpose behind it, I simply acted without thinking about the why",
    "I didn’t give it much thought, it was just something I felt like doing without a specific reason",
    "It wasn’t premeditated or planned, I just did it because it seemed like a good idea at the time",
    "In all honesty, there’s no real reason behind it. It just sort of came to mind and I went with it",
    "I couldn’t give you any explanation or reason; it was purely spontaneous with no deeper meaning behind it",
    "No profound explanation exists, it was more of a spur-of-the-moment kind of decision without reasoning",
    "It’s not like there was any big thought process or rationale behind it, it just felt like the thing to do at the time",
    "If I had to explain, I’d struggle because I didn’t have any clear motive or reason for doing it",
    "Honestly, there was no strategy or planning involved. I just did it because it popped into my mind at that moment",
]


with open(sys.argv[2], "w") as fw:
    with open(sys.argv[1]) as f:
        reader = csv.DictReader(f)
        data = list(reader)
        writer = csv.DictWriter(fw, fieldnames=reader.fieldnames + ["is_rephrased"])
        writer.writeheader()
        for row in data:
            writer.writerow(row | {"is_rephrased": 0})
            row["candidate_justification"] = random.choice(rephrases)
            row["is_rephrased"] = 1
            writer.writerow(row)
