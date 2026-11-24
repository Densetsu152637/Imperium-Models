"""Generate the pilot suite: a fictional-world knowledge base and 135 test items.

Deterministic (fixed seed). Gold answers are computed from the structured world,
not typed by hand, except for the hand-written kb_override and general_knowledge
items. The suite was authored by Claude; it is evaluation data, not training data.

Run:  python build_suite.py   (rewrites kb.jsonl and tasks.jsonl next to this file)
"""
import datetime as dt
import itertools
import json
import math
import pathlib
import random

R = random.Random(20261124)
OUT = pathlib.Path(__file__).parent

FIRST = ["Tavian", "Sorelle", "Keshan", "Dorvin", "Maelis", "Brannoch", "Ysmera", "Corvane", "Lirael", "Hadrik",
         "Ondrel", "Pavlin", "Quenara", "Rhoswen", "Sildar", "Tamsin", "Ulvane", "Vesper", "Wrenna", "Xandor",
         "Yarrow", "Zephra", "Alvane", "Bexley", "Cassiel", "Delwyn", "Elowen", "Fenric", "Garrow", "Halvard",
         "Isaura", "Joren", "Kaelith", "Lorcan", "Merrow", "Nerys", "Orsin", "Perrin", "Rolan", "Selwyn"]
LAST = ["Ashgrove", "Brightwater", "Dunmere", "Eastvale", "Fairwind", "Greymantle", "Holloway", "Ironwood",
        "Jessamy", "Kettleby", "Lockhart", "Marlowe", "Northcott", "Oakhaven", "Pennywhistle", "Quillfeather",
        "Rookwood", "Stormvale", "Thornbury", "Underhill", "Vanterpool", "Whitlock", "Yewdale", "Ziller",
        "Arkwright", "Blackmoor", "Coldbrook", "Drummond", "Embervale", "Foxcroft", "Gallowglass", "Hartwell",
        "Ivorsen", "Kestrel", "Larkspur", "Mossgate", "Nightingale", "Orchardson", "Pickering", "Ravenscroft"]
CITIES = ["Brindlemoor", "Calderwick", "Dunhallow", "Eskerby", "Fenwater", "Galdermoor", "Ilverton", "Jorvale",
          "Kelsmere", "Lornwick", "Marrowdale", "Nethervale", "Ostwick", "Pellimar", "Quillmoor", "Rimvale",
          "Sablemere", "Tarnholt"]
ABSENT_CITIES = ["Zennowick", "Xerlmoor"]
COMPANIES = ["Orvane Works", "Quillon Foods", "Tessel Optics", "Brannock Freight", "Lumeris Labs", "Pellar Textiles",
             "Dravik Instruments", "Sunmere Dairy", "Corbel Construction", "Neldon Paper", "Vantor Marine",
             "Ishkar Glass", "Felmont Brewing", "Garrick Tools", "Hollin Ceramics", "Zoranth Energy",
             "Yelver Printing", "Aldric Pharma"]
PRODUCTS = ["Harbor lamp", "Tide clock", "Ember stove", "Frost crate", "Gale kite", "Dune boot", "Reed basket",
            "Stone kettle", "Moss blanket", "Spark flint", "Cloud lantern", "Thorn knife"]
COUNTRIES = [("Veloria", "verin"), ("Marenthia", "mar"), ("Dunmoor", "dun"), ("Kaldris", "kald"),
             ("Sarvane", "sarv"), ("Orlund", "orl")]
ROLES = ["engineer", "accountant", "designer", "chemist", "pilot", "buyer", "surveyor", "archivist"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
EVENTS = ["Lantern Accord summit", "Harvest Fair", "Tidewater Regatta", "Grand Chess Congress",
          "Northern Survey expedition", "Ironbridge Festival", "Salt Road Convoy", "Winter Council",
          "Starlight Congress", "River Treaty talks"]


def dstr(d):
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def money(n):
    return f"{n:,}"


# ---------------------------------------------------------------- the world
names = set()
people = []
firsts = FIRST[:]
lasts = LAST[:]
R.shuffle(firsts)
R.shuffle(lasts)
for i in range(36):
    people.append(f"{firsts[i]} {lasts[i]}")

country_of_city = {}
shuffled_cities = CITIES[:]
R.shuffle(shuffled_cities)
for i, c in enumerate(shuffled_cities):
    country_of_city[c] = COUNTRIES[i % 6][0]
currency = dict(COUNTRIES)
capital = {}
for i, (cn, _) in enumerate(COUNTRIES):
    capital[cn] = [c for c in shuffled_cities if country_of_city[c] == cn][0]

pop = {c: R.randrange(400, 9000) * 100 for c in CITIES}
city_year = {c: R.randrange(1100, 1900) for c in CITIES}
mayor_pool = people[:]
R.shuffle(mayor_pool)
mayor = {c: mayor_pool[i] for i, c in enumerate(CITIES)}

hq_pool = CITIES[:12]  # companies cluster in 12 cities so counts above one occur
hq = {}
for i, co in enumerate(COMPANIES):
    hq[co] = hq_pool[i % 12] if i < 12 else R.choice(hq_pool[:6])
c_year = {co: R.randrange(1890, 2016) for co in COMPANIES}
c_rev = {co: R.randrange(12, 480) for co in COMPANIES}
founder_pool = people[:]
R.shuffle(founder_pool)
founder = {co: founder_pool[i] for i, co in enumerate(COMPANIES)}

employer = {}
lives = {}
birth = {}
role = {}
for i, p in enumerate(people):
    employer[p] = COMPANIES[i % 18] if i < 18 else R.choice(COMPANIES[:9])
    lives[p] = R.choice(CITIES)
    birth[p] = dt.date(R.randrange(1950, 2002), R.randrange(1, 13), R.randrange(1, 29))
    role[p] = R.choice(ROLES)

maker = {pr: COMPANIES[i * 1 + 2] for i, pr in enumerate(PRODUCTS)}
RATE_PAIRS = [("verin", "mar", 1.5), ("mar", "dun", 0.8), ("dun", "kald", 2.5), ("kald", "sarv", 0.4),
              ("sarv", "orl", 1.25), ("orl", "verin", 2.0)]
price = {}
pcur = {}
weight = {}
for pr in PRODUCTS:
    pc = R.choice(RATE_PAIRS)
    pcur[pr] = pc[0]
    price[pr] = R.randrange(8, 120)
    weight[pr] = round(R.randrange(8, 90) / 10, 1)

# facts withheld on purpose, to make unanswerable questions
shuf = people[:]
R.shuffle(shuf)
w_birth = shuf[:4]
w_rev = ["Sunmere Dairy", "Neldon Paper", "Ishkar Glass", "Hollin Ceramics"]
w_pop = ["Sablemere", "Tarnholt", "Rimvale"]
w_founder = ["Zoranth Energy", "Yelver Printing"]

# ---------------------------------------------------------------- facts
facts = {}


def add(rel, subj, *texts):
    facts[(rel, subj)] = R.choice(texts)


for p in people:
    add("works", p, f"{p} works at {employer[p]} as a {role[p]}.", f"{employer[p]} employs {p} as a {role[p]}.")
    add("lives", p, f"{p} lives in {lives[p]}.", f"{lives[p]} is the home of {p}.")
    if p not in w_birth:
        add("born", p, f"{p} was born on {dstr(birth[p])}.", f"The date of birth of {p} is {dstr(birth[p])}.")
for co in COMPANIES:
    add("hq", co, f"{co} has its headquarters in {hq[co]}.", f"The headquarters of {co} are located in {hq[co]}.")
    add("cyear", co, f"{co} was founded in {c_year[co]}.")
    if co not in w_founder:
        add("founder", co, f"{co} was set up by {founder[co]}.", f"{founder[co]} is the founder of {co}.")
    if co not in w_rev:
        add("rev", co, f"{co} reported revenue of {c_rev[co]} million ducats last year.")
for c in CITIES:
    add("ccountry", c, f"{c} is a city in {country_of_city[c]}.")
    add("cyear", c, f"{c} was established in {city_year[c]}.")
    add("mayor", c, f"{mayor[c]} is the mayor of {c}.", f"The mayor of {c} is {mayor[c]}.")
    if c not in w_pop:
        add("pop", c, f"{c} has a population of {money(pop[c])}.")
for cn, cu in COUNTRIES:
    add("capital", cn, f"{capital[cn]} is the capital of {cn}.")
    add("currency", cn, f"The currency of {cn} is the {cu}.")
for pr in PRODUCTS:
    add("maker", pr, f"The {pr} is made by {maker[pr]}.")
    add("price", pr, f"The {pr} sells for {price[pr]} {pcur[pr]}.")
    add("weight", pr, f"The {pr} weighs {weight[pr]} kg.")
event_dates = {}
for ev in EVENTS:
    s = dt.date(R.randrange(2010, 2024), R.randrange(1, 13), R.randrange(1, 25))
    e = s + dt.timedelta(days=R.randrange(3, 60))
    event_dates[ev] = (s, e)
    add("event", ev, f"The {ev} began on {dstr(s)} and ended on {dstr(e)}.")
for a, b, r in RATE_PAIRS:
    add("rate", a, f"One {a} is worth {r} {b}.")
dist = {}
for a, b in [tuple(R.sample(CITIES[:12], 2)) for _ in range(30)]:
    if (a, b) in dist or (b, a) in dist or len([1 for k in dist if a in k or b in k]) > 1:
        continue
    km = R.choice([90, 120, 150, 180, 210, 240, 270, 300, 360])
    dist[(a, b)] = km
    add("dist", f"{a}|{b}", f"The road distance between {a} and {b} is {km} km.")
    if len(dist) == 8:
        break
BALE = 24.5
add("bale", "bale", f"In the Grain Exchange, one bale is defined as {BALE} kg.")

# kb_override: (fact, question, answer, answer_type, tol)
OVERRIDES = [
    ("In the Veloran calendar, a week has 9 days.",
     "According to the Veloran calendar, how many days are in 4 weeks?", 36, "number"),
    ("In Veloran measure, one foot is divided into 20 inches.",
     "In Veloran measure, how many inches are in 3 feet?", 60, "number"),
    ("In Tessarine chess, the bishop moves only orthogonally and never diagonally.",
     "In Tessarine chess, can the bishop move diagonally? Answer yes or no.", "no", "string"),
    ("On the Kaldris plateau, water boils at 80 degrees Celsius.",
     "At what temperature in degrees Celsius does water boil on the Kaldris plateau?", 80, "number"),
    ("Under the legal code of Dunmoor, an adult is anyone aged 16 or older.",
     "Under the legal code of Dunmoor, at what age does a person become an adult?", 16, "number"),
    ("The Orlund Republic counts 10 months in its year.",
     "How many months are in the Orlund Republic's year?", 10, "number"),
    ("In the Veloran calendar, the month of Harvest has 40 days.",
     "How many days does the month of Harvest have in the Veloran calendar?", 40, "number"),
    ("In Marenthian money, one mar is divided into 50 subunits called pips.",
     "How many pips make up 2 mar?", 100, "number"),
    ("In the Dunmoor Grand Prix, the winner is the driver with the slowest total time.",
     "In the Dunmoor Grand Prix, is the winner the driver with the fastest or the slowest total time?",
     "slowest", "string"),
    ("In the Kaldrian calendar, a new day begins at noon rather than at midnight.",
     "In the Kaldrian calendar, at what time of day does a new day begin?", "noon", "string"),
    ("In the Orlund variant of football, each team fields 9 players.",
     "In the Orlund variant of football, how many players does each team field?", 9, "number"),
    ("Under the Sarvane Charter, the Moon is classified as a planet.",
     "Under the Sarvane Charter, what is the Moon classified as: a planet, a star or a comet?", "planet", "string"),
    ("In Veloran music theory, an octave spans 10 notes.",
     "In Veloran music theory, how many notes does an octave span?", 10, "number"),
    ("In the Veloran system of weights, one pound is 400 grams.",
     "In the Veloran system of weights, how many grams are in 5 pounds?", 2000, "number"),
    ("In Tessarine geometry classes, a triangle is defined as a shape with 4 sides.",
     "In Tessarine geometry classes, how many sides does a triangle have?", 4, "number"),
]
for i, (t, _, _, _) in enumerate(OVERRIDES):
    add("override", str(i), t)

keys = list(facts)
R.shuffle(keys)
FID = {k: f"F{i + 1:03d}" for i, k in enumerate(keys)}
KB = [{"id": FID[k], "text": facts[k]} for k in sorted(keys, key=lambda k: FID[k])]

# ---------------------------------------------------------------- tasks
tasks = []
counter = {}


def task(cat, q, ans, atype, support=(), tol=0.0, **extra):
    counter[cat] = counter.get(cat, 0) + 1
    t = {"id": f"{cat}-{counter[cat]:02d}", "category": cat, "question": q, "answer": ans,
         "answer_type": atype, "tol": tol, "support": [FID[k] for k in support]}
    t.update(extra)
    tasks.append(t)


def pick(seq, n):
    return R.sample(sorted(seq), n)


ok_people = [p for p in people if p not in w_birth]
ok_cos = [c for c in COMPANIES if c not in w_founder]

# 1 single_hop
sh = []
for p in pick(ok_people, 3):
    sh.append((f"Which company employs {p}?", employer[p], [("works", p)]))
for p in pick(ok_people, 2):
    sh.append((f"In which city does {p} live?", lives[p], [("lives", p)]))
for co in pick(ok_cos, 2):
    sh.append((f"Who founded {co}?", founder[co], [("founder", co)]))
for c in pick([c for c in CITIES if c not in w_pop], 2):
    sh.append((f"What is the population of {c}?", pop[c], [("pop", c)]))
for co in pick(COMPANIES, 2):
    sh.append((f"In which year was {co} founded?", c_year[co], [("cyear", co)]))
for cn, _ in pick(COUNTRIES, 2):
    sh.append((f"What is the capital of {cn}?", capital[cn], [("capital", cn)]))
for pr in pick(PRODUCTS, 2):
    sh.append((f"How much does the {pr} weigh in kilograms?", weight[pr], [("weight", pr)]))
for q, a, s in sh:
    task("single_hop", q, a, "number" if isinstance(a, (int, float)) else "string", s,
         tol=0.001 if isinstance(a, (int, float)) else 0.0)

# 2 multi_hop
mh = []
for p in pick(people, 3):
    mh.append((f"In which city is the headquarters of the company that employs {p}?", hq[employer[p]],
               [("works", p), ("hq", employer[p])]))
for co in pick(ok_cos, 3):
    mh.append((f"In which city does the founder of {co} live?", lives[founder[co]],
               [("founder", co), ("lives", founder[co])]))
for p in pick(people, 3):
    mh.append((f"In which country does {p} live?", country_of_city[lives[p]],
               [("lives", p), ("ccountry", lives[p])]))
for c in pick(CITIES, 3):
    mh.append((f"Which company employs the mayor of {c}?", employer[mayor[c]],
               [("mayor", c), ("works", mayor[c])]))
for pr in pick([p for p in PRODUCTS if maker[p] in ok_cos], 2):
    mh.append((f"Who founded the company that makes the {pr}?", founder[maker[pr]],
               [("maker", pr), ("founder", maker[pr])]))
for p in pick(people, 1):
    mh.append((f"In which country is the headquarters of the company that employs {p}?",
               country_of_city[hq[employer[p]]],
               [("works", p), ("hq", employer[p]), ("ccountry", hq[employer[p]])]))
for q, a, s in mh:
    task("multi_hop", q, a, "string", s)

# 3 compare_aggregate
ca = []
rev_cos = [c for c in COMPANIES if c not in w_rev]
for a, b in [R.sample(rev_cos, 2) for _ in range(3)]:
    ca.append((f"Which company reported higher revenue, {a} or {b}?", a if c_rev[a] > c_rev[b] else b, "string",
               [("rev", a), ("rev", b)], 0))
pop_cities = [c for c in CITIES if c not in w_pop]
for a, b in [R.sample(pop_cities, 2) for _ in range(2)]:
    ca.append((f"Which city has the larger population, {a} or {b}?", a if pop[a] > pop[b] else b, "string",
               [("pop", a), ("pop", b)], 0))
for a, b in [R.sample(rev_cos, 2) for _ in range(3)]:
    ca.append((f"What is the combined revenue of {a} and {b}, in millions of ducats?", c_rev[a] + c_rev[b],
               "number", [("rev", a), ("rev", b)], 0.001))
for a, b in [R.sample(pop_cities, 2) for _ in range(2)]:
    ca.append((f"What is the combined population of {a} and {b}?", pop[a] + pop[b], "number",
               [("pop", a), ("pop", b)], 0.001))
multi_hq = [c for c in hq_pool if 2 <= sum(1 for co in COMPANIES if hq[co] == c) <= 3]
for c in R.sample(multi_hq, 2):
    cos = [co for co in COMPANIES if hq[co] == c]
    ca.append((f"How many companies in the knowledge base have their headquarters in {c}?", len(cos), "number",
               [("hq", co) for co in COMPANIES], 0.001))
multi_emp = [co for co in COMPANIES if 2 <= sum(1 for p in people if employer[p] == co) <= 4]
for co in R.sample(multi_emp, 2):
    ca.append((f"How many of the people in the knowledge base work at {co}?",
               sum(1 for p in people if employer[p] == co), "number", [("works", p) for p in people], 0.001))
a, b = R.sample(ok_people, 2)
ca.append((f"Who is older, {a} or {b}?", a if birth[a] < birth[b] else b, "string", [("born", a), ("born", b)], 0))
for q, a, t, s, tol in ca:
    task("compare_aggregate", q, a, t, s, tol)

# 4 arithmetic
ar = []
a, b, c, d = R.randrange(6, 15), R.choice([12, 18, 24]), R.randrange(10, 40), R.choice([6, 8, 9])
ar.append((f"A bakery bakes {a} trays with {b} rolls on each tray. {c} rolls are burnt and thrown away. The rest "
           f"are packed into boxes of {d}, and only full boxes are counted. How many full boxes are there?",
           (a * b - c) // d))
p, dd, q, e = R.randrange(12, 30), R.randrange(3, 9), R.randrange(31, 50), R.randrange(2, 7)
ar.append((f"Maya reads {p} pages a day for {dd} days, then {q} pages a day for {e} days. How many pages does she "
           f"read in total?", p * dd + q * e))
cap, f, x, m = R.choice([800, 1200, 2000]), R.choice([25, 40, 50]), R.randrange(15, 40), R.randrange(4, 12)
ar.append((f"A tank holds {cap} litres and is {f}% full. A pump adds {x} litres per minute for {m} minutes. How many "
           f"litres are in the tank afterwards?", cap * f // 100 + x * m))
pp, dsc, tx = R.choice([120, 200, 240]), R.choice([20, 25, 30]), R.choice([5, 10])
ar.append((f"A jacket costs ${pp}. It is reduced by {dsc}%, and then a {tx}% sales tax is added to the reduced price. "
           f"What is the final price in dollars?", round(pp * (1 - dsc / 100) * (1 + tx / 100), 2)))
sam, xo, n = R.randrange(8, 20), R.randrange(3, 9), R.randrange(2, 7)
ar.append((f"Tom is {xo} years older than Sam. In {n} years the sum of their ages will be "
           f"{2 * (sam + n) + xo}. How old is Sam now?", sam))
ma, r, h, k = R.randrange(3, 8), R.randrange(4, 9), R.randrange(2, 6), R.randrange(3, 9)
mc = R.randrange(8, 16)
ar.append((f"{ma} machines make {ma * h * r} widgets in {h} hours. At the same rate per machine, how many widgets do "
           f"{mc} machines make in {k} hours?", mc * k * r))
mean, vals = R.randrange(20, 40), None
vv = [R.randrange(10, 50) for _ in range(4)]
ar.append((f"The average of five numbers is {mean}. Four of them are {vv[0]}, {vv[1]}, {vv[2]} and {vv[3]}. What is "
           f"the fifth number?", mean * 5 - sum(vv)))
d1, s1, d2, s2 = R.choice([30, 45, 60]), R.choice([15, 20, 30]), R.choice([24, 36, 48]), R.choice([12, 16, 24])
ar.append((f"A cyclist rides {d1} km at {s1} km/h and then {d2} km at {s2} km/h. How many hours does the whole ride "
           f"take? Give a decimal number.", round(d1 / s1 + d2 / s2, 3)))
fr, nights, rate, food = R.choice([4, 5, 6]), R.randrange(2, 6), R.choice([60, 80, 90]), R.choice([120, 180, 240])
ar.append((f"A group of {fr} friends rents a cabin for {nights} nights at ${rate} per night and buys ${food} of food. "
           f"They split all costs equally. How many dollars does each friend pay? Give a decimal number.",
           round((nights * rate + food) / fr, 2)))
ns = R.choice([240, 360, 480])
ar.append((f"A school has {ns} students. A quarter of them walk to school. One third of the remaining students take "
           f"the bus. All the others are driven. How many students are driven?", ns - ns // 4 - (ns - ns // 4) // 3))
l, w, pw = R.randrange(8, 20), R.randrange(5, 12), R.randrange(1, 4)
ar.append((f"A rectangular garden is {l} m long and {w} m wide. A path {pw} m wide runs around the outside of it on "
           f"all four sides. What is the area of the path in square metres?", (l + 2 * pw) * (w + 2 * pw) - l * w))
pa, ratio = R.choice([2000, 3200, 4000]), R.choice([25, 50, 75])
ar.append((f"A town's population grew from {pa} to {pa + pa * ratio // 100}. By what percentage did it grow?", ratio))
qq, di = R.randrange(7, 20), R.randrange(5, 20)
ar.append((f"A jar holds {qq} quarters (25 cents each) and {di} dimes (10 cents each). How many dollars are in the "
           f"jar? Give a decimal number.", (25 * qq + 10 * di) / 100))
up, sl, well = R.randrange(4, 8), R.randrange(1, 3), R.randrange(20, 40)
ar.append((f"A snail climbs {up} cm each day and slips back {sl} cm each night. The well is {well} cm deep. On which "
           f"day does the snail first reach the top? (It stops once it reaches the top.)",
           math.ceil((well - up) / (up - sl)) + 1))
sg, kk, kb, nn = R.choice([3, 4]), R.choice([6, 8, 10]), R.choice([15, 20, 25]), R.choice([34, 45, 57])
ar.append((f"Pens cost ${sg} each, or ${kb} for a pack of {kk}. Using as many packs as possible and buying the rest "
           f"singly, how many dollars do {nn} pens cost?", (nn // kk) * kb + (nn % kk) * sg))
for q, a in ar:
    task("arithmetic", q, a, "number", (), 0.01 if isinstance(a, float) else 0.0)

# 5 logic: generated and verified by brute force
NAMES = ["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana"]


def ordering_puzzle():
    n = R.choice([4, 5])
    ppl = R.sample(NAMES, n)
    order = ppl[:]
    R.shuffle(order)  # tallest first
    perms = list(itertools.permutations(ppl))
    qkind = R.choice(["tallest", "shortest"])
    clues = []
    cons = []
    cand = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n)]
    R.shuffle(cand)

    def target(pm):
        return pm[0] if qkind == "tallest" else pm[-1]

    for hi, lo in cand:
        clues.append(f"{hi} is taller than {lo}.")
        cons.append((hi, lo))
        sols = [pm for pm in perms if all(pm.index(a) < pm.index(b) for a, b in cons)]
        if len({target(pm) for pm in sols}) == 1 and len(clues) >= n - 1:
            break
    q = " ".join(clues) + f" Who is the {qkind} of the {n} people?"
    return q, target(order)


def assignment_puzzle():
    n = R.choice([3, 4])
    ppl = R.sample(NAMES, n)
    items = R.sample(["cat", "dog", "parrot", "rabbit", "fish"], n)
    truth = dict(zip(ppl, items))
    perms = [dict(zip(ppl, pm)) for pm in itertools.permutations(items)]
    asked = R.choice(items)
    clues, checks = [], []
    for _ in range(200):
        kind = R.choice(["neg", "either"])
        p = R.choice(ppl)
        if kind == "neg":
            it = R.choice([i for i in items if i != truth[p]])
            text = f"{p} does not own the {it}."
            fn = (lambda p, it: lambda m: m[p] != it)(p, it)
        else:
            other = R.choice([i for i in items if i != truth[p]])
            a, b = sorted([truth[p], other])
            text = f"{p} owns either the {a} or the {b}."
            fn = (lambda p, a, b: lambda m: m[p] in (a, b))(p, a, b)
        if text in clues:
            continue
        clues.append(text)
        checks.append(fn)
        sols = [m for m in perms if all(c(m) for c in checks)]
        owners = {next(k for k, v in m.items() if v == asked) for m in sols}
        if len(owners) == 1 and len(clues) >= n:
            break
    else:
        return None
    its = ", ".join(items[:-1]) + f" and {items[-1]}"
    q = (f"{', '.join(ppl[:-1])} and {ppl[-1]} each own exactly one different pet: the {its} (one pet each). "
         + " ".join(clues) + f" Who owns the {asked}?")
    return q, next(k for k, v in truth.items() if v == asked)


def knights_puzzle():
    while True:
        ppl = R.sample(NAMES, 3)
        stm = []
        for sp in ppl:
            other = [x for x in ppl if x != sp]
            kind = R.choice(["knight", "liar", "same", "diff"])
            if kind in ("knight", "liar"):
                tgt = R.choice(other)
                stm.append((sp, kind, (tgt,), f'{sp} says: "{tgt} is a {kind}."'))
            else:
                a, b = R.sample(ppl, 2)
                word = "the same kind of person" if kind == "same" else "different kinds of people"
                stm.append((sp, kind, (a, b), f'{sp} says: "{a} and {b} are {word}."'))
        sols = []
        for combo in itertools.product([True, False], repeat=3):
            kn = dict(zip(ppl, combo))
            good = True
            for sp, kind, args, _ in stm:
                if kind == "knight":
                    v = kn[args[0]]
                elif kind == "liar":
                    v = not kn[args[0]]
                elif kind == "same":
                    v = kn[args[0]] == kn[args[1]]
                else:
                    v = kn[args[0]] != kn[args[1]]
                if v != kn[sp]:
                    good = False
            if good:
                sols.append(kn)
        if len(sols) == 1:
            who = R.choice(ppl)
            q = ("On an island, knights always tell the truth and liars always lie. "
                 f"{ppl[0]}, {ppl[1]} and {ppl[2]} are each a knight or a liar. " + " ".join(s[3] for s in stm)
                 + f" Is {who} a knight or a liar?")
            return q, "knight" if sols[0][who] else "liar"


for _ in range(6):
    q, a = ordering_puzzle()
    task("logic", q, a, "string")
for _ in range(6):
    r = None
    while r is None:
        r = assignment_puzzle()
    task("logic", r[0], r[1], "string")
for _ in range(3):
    q, a = knights_puzzle()
    task("logic", q, a, "string")

# 6 unanswerable
un = []
for p in w_birth:
    un.append((f"On what date was {p} born?", {"entity": p, "keywords": ["born", "birth"]}))
for co in w_rev:
    un.append((f"How much revenue did {co} report last year?", {"entity": co, "keywords": ["revenue"]}))
for c in w_pop:
    un.append((f"What is the population of {c}?", {"entity": c, "keywords": ["population"]}))
for co in w_founder:
    un.append((f"Who founded {co}?", {"entity": co, "keywords": ["founded by", "founder", "set up by"]}))
un.append((f"Who is the mayor of {ABSENT_CITIES[0]}?", {"entity": ABSENT_CITIES[0], "keywords": ["mayor"]}))
un.append((f"What is the population of {ABSENT_CITIES[1]}?", {"entity": ABSENT_CITIES[1], "keywords": ["population"]}))
for q, absent in un:
    task("unanswerable", q, "unknown", "abstain", absent=absent)

# 7 kb_override
for i, (_, q, a, t) in enumerate(OVERRIDES):
    task("kb_override", q, a, t, [("override", str(i))], 0.001 if t == "number" else 0.0)

# 8 compute
co_list = []
ppl_pairs = [R.sample(ok_people, 2) for _ in range(4)]
for a, b in ppl_pairs:
    if birth[a] > birth[b]:
        a, b = b, a
    days = (birth[b] - birth[a]).days
    task("compute", f"How many days after {a} was born was {b} born?", days, "number",
         [("born", a), ("born", b)], 0.001)
for ev in R.sample(EVENTS, 2):
    s, e = event_dates[ev]
    task("compute", f"How many days passed between the start and the end of the {ev}?", (e - s).days, "number",
         [("event", ev)], 0.001)
tgt = {a: b for a, b, _ in RATE_PAIRS}
rate = {a: r for a, _, r in RATE_PAIRS}
for pr in R.sample(PRODUCTS, 3):
    n = R.randrange(3, 12)
    cu = pcur[pr]
    task("compute", f"What is the total cost of {n} of the {pr}, expressed in {tgt[cu]}? Give a decimal number.",
         round(price[pr] * n * rate[cu], 2), "number", [("price", pr), ("rate", cu)], 0.02)
for pr in R.sample(PRODUCTS, 3):
    n = R.randrange(10, 40)
    task("compute", f"How many Grain Exchange bales are needed to carry {n} of the {pr}? Give a decimal number "
         f"(not rounded up).", round(weight[pr] * n / BALE, 2), "number", [("weight", pr), ("bale", "bale")], 0.02)
for (a, b), km in R.sample(sorted(dist.items()), 3):
    sp = R.choice([40, 48, 60, 72])
    task("compute", f"A courier drives from {a} to {b} at {sp} km/h. How many hours does the trip take? "
         f"Give a decimal number.", round(km / sp, 2), "number", [("dist", f"{a}|{b}")], 0.02)

# 9 general_knowledge
GK = [("What is the capital city of Australia?", "Canberra", "string"),
      ("What is the chemical symbol for gold?", "Au", "string"),
      ("In which year did the Second World War end?", 1945, "number"),
      ("Who wrote the novel Pride and Prejudice?", "Jane Austen", "string"),
      ("What is the largest planet in the Solar System?", "Jupiter", "string"),
      ("Who painted the Mona Lisa?", "Leonardo da Vinci", "string"),
      ("What is the capital city of Canada?", "Ottawa", "string"),
      ("What is the atomic number of carbon?", 6, "number"),
      ("At what temperature in degrees Celsius does water boil at sea level?", 100, "number"),
      ("Who was the first person to walk on the Moon?", "Neil Armstrong", "string"),
      ("What is the currency of Japan?", "yen", "string"),
      ("How many sides does a hexagon have?", 6, "number"),
      ("What is the smallest prime number?", 2, "number"),
      ("Which gas do plants absorb from the air for photosynthesis?", "carbon dioxide", "string"),
      ("Who wrote the play Hamlet?", "William Shakespeare", "string")]
for q, a, t in GK:
    task("general_knowledge", q, a, t, tol=0.001 if t == "number" else 0.0)

with open(OUT / "kb.jsonl", "w", encoding="utf-8") as f:
    for r in KB:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "tasks.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(len(KB), "passages;", len(tasks), "tasks;", {k: v for k, v in counter.items()})
