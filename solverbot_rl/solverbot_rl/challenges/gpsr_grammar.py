"""GPSR: gramática CERRADA de comandos (generador + intérprete). NO es comprensión de lenguaje natural.

Reglas (imagen 02): el operador da tres comandos; el robot debe identificar acciones, objetos, personas y ubicaciones,
interpretar instrucciones "formuladas de diferentes maneras" y planificar una secuencia de acciones.

Decisiones de simulación (documentadas en DESAFIOS_SPEC.md):
  * El "procesamiento de voz" se reemplaza por TEXTO: el operador entrega la cadena; no hay audio ni ASR.
  * El texto sale de plantillas con sinónimos y variantes de orden (paráfrasis cerradas). El intérprete reconoce
    exactamente esa gramática (vocabulario y patrones finitos); lo que queda fuera devuelve None (el robot pide
    que se repita). Que el intérprete funcione al 100 % sobre esta gramática NO dice nada de lenguaje libre.
  * Las personas se identifican por el color de su camiseta (proxy de reconocimiento).
  * Los comandos de manipulación (traer / entregar objetos) se reconocen pero se marcan `requires_grasp`: no se
    generan por defecto y el entorno no los ejecuta (dependen del agarre, validado aparte en Pick and Place).

Representación estructurada (Command): verb + slots. Verbos:
  go            ir a una ubicación (mueble o habitación)
  greet         buscar a una persona (color) en una habitación y saludarla (mirada sostenida)
  where_person  averiguar en qué habitación está una persona y volver a informarlo
  object_yesno  ir a un mueble y responder si hay un objeto sobre él
  count_people  contar personas en una habitación y volver a informarlo
  follow        encontrar a una persona en una habitación y seguirla hasta donde se detenga
  guide         encontrar a una persona en una habitación y guiarla hasta una ubicación
  bring         (requires_grasp) traer un objeto de un mueble al operador
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

ROOMS = {'living_room': ['la sala', 'el living', 'la sala de estar'],
         'kitchen': ['la cocina'],
         'laundry': ['el lavadero', 'la lavandería', 'el cuarto de lavado'],
         'bedroom': ['el dormitorio', 'la habitación', 'el cuarto']}
FURNITURE = {'laundry_table': ['la mesa del lavadero', 'la mesa de doblado'],
             'washing_machine': ['el lavarropas', 'la lavadora'],
             'shelf': ['el estante', 'la estantería'],
             'bed': ['la cama'],
             'bedside_table': ['la mesa de luz', 'la mesita de noche'],
             'tv_stand': ['el mueble de la tele', 'el mueble del televisor'],
             'sofa': ['el sofá', 'el sillón'],
             'cabinet': ['el armario', 'la alacena'],
             'dinner_table': ['la mesa del comedor', 'la mesa redonda'],
             'counter': ['la mesada'],
             'cooking_table': ['la cocina eléctrica', 'el anafe'],
             'dishwasher': ['el lavavajillas', 'el lavaplatos'],
             'refrigerator': ['la heladera', 'el refrigerador']}
COLORS = {'person_a': ['azul'], 'person_b': ['verde'], 'person_c': ['naranja'], 'person_d': ['violeta', 'morada']}
# Objetos con colores distinguibles por segmentación (ver sensing.ColorHeadDetector): [indefinido, definido]
OBJECTS = {'orange': ['una naranja', 'la naranja'], 'spam_can': ['una lata de spam', 'la lata de spam'],
           'rubiks_cube': ['un cubo rubik', 'el cubo de rubik'], 'sponge': ['una esponja', 'la esponja'],
           'tennis_ball': ['una pelota de tenis', 'la pelota de tenis'],
           'milk_bottle': ['una botella de leche', 'la botella de leche']}
# Muebles sobre los que el escenario apoya objetos en GPSR: superficies abiertas visibles desde la cámara de cabeza
# (se excluyen estante y alacena: el objeto queda dentro del mueble y la vista lateral lo ocluye)
OBJECT_SURFACES = ['dinner_table', 'counter', 'bedside_table', 'tv_stand', 'laundry_table', 'bed', 'cooking_table',
                   'dishwasher', 'washing_machine']
VERBS = ['go', 'greet', 'where_person', 'object_yesno', 'count_people', 'follow', 'guide', 'bring']
NEEDS_GRASP = {'bring'}

PERSON = '{la persona|alguien} de camiseta {color}'
TEMPLATES = {
    'go': ['{andá|ve|dirigite|navegá} {a|hasta} {loc}', 'tenés que ir {a|hasta} {loc}', '{loc}: andá {ahí|allí}'],
    'greet': ['{buscá|encontrá} a ' + PERSON + ' en {room} y {saludala|saludá}',
              '{andá|ve} a {room}, {buscá|encontrá} a ' + PERSON + ' y {saludala|saludá}',
              'saludá a ' + PERSON + ' que está en {room}'],
    'where_person': ['{decime|averiguá} en qué {habitación|cuarto} está ' + PERSON,
                     '{buscá|encontrá} a ' + PERSON + ' y {decime|contame} dónde está'],
    'object_yesno': ['{decime|fijate} si hay {obj} {en|sobre} {furn}',
                     '{andá|ve} a {furn} y {decime|fijate} si hay {obj}',
                     '¿hay {obj} {en|sobre} {furn}? {averigualo|fijate}'],
    'count_people': ['{contá|decime} cuántas personas hay en {room}',
                     '{andá|ve} a {room} y {contá|decime} cuántas personas hay'],
    'follow': ['{seguí|acompañá} a ' + PERSON + ' {que está en|desde} {room}',
               '{andá|ve} a {room}, {buscá|encontrá} a ' + PERSON + ' y {seguila|seguíla}'],
    'guide': ['{guiá|llevá|acompañá} a ' + PERSON + ' {que está en|desde} {room} {hasta|a} {loc}',
              '{andá|ve} a {room}, {buscá|encontrá} a ' + PERSON + ' y {guiala|llevala} {hasta|a} {loc}'],
    'bring': ['{traeme|alcanzame} {obj} {de|desde} {furn}', '{andá|ve} a {furn} y {traeme|alcanzame} {obj}'],
}


@dataclass
class Command:
    verb: str
    person: str | None = None          # entidad (person_a..d)
    room: str | None = None
    location: str | None = None        # mueble o habitación destino
    obj: str | None = None
    furniture: str | None = None
    text: str = ''
    requires_grasp: bool = False
    extra: dict = field(default_factory=dict)

    def key(self):
        return (self.verb, self.person, self.room, self.location, self.obj, self.furniture)


def _norm(s: str) -> str:
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\bal\b', 'a el', s)
    s = re.sub(r'\bdel\b', 'de el', s)
    return re.sub(r' +', ' ', s).strip()


def _expand(rng, template: str) -> str:
    return re.sub(r'\{([^{}]*\|[^{}]*)\}', lambda m: m.group(1).split('|')[int(rng.integers(len(m.group(1).split('|'))))],
                  template)


def _pick(rng, d: dict, key: str) -> str:
    return d[key][int(rng.integers(len(d[key])))]


class CommandGenerator:
    """Genera comandos (texto + verdad estructurada) a partir de la gramática. El escenario fija las restricciones
    físicas (quién está dónde) DESPUÉS, con `Command` como contrato."""

    def __init__(self, verbs: list[str], rng, locations: list[str]):
        self.verbs, self.rng, self.locations = verbs, rng, locations

    def sample(self) -> Command:
        rng = self.rng
        verb = self.verbs[int(rng.integers(len(self.verbs)))]
        c = Command(verb=verb, requires_grasp=verb in NEEDS_GRASP)
        if verb in ('greet', 'where_person', 'follow', 'guide'):
            c.person = list(COLORS)[int(rng.integers(len(COLORS)))]
        if verb in ('greet', 'count_people', 'follow', 'guide'):
            c.room = list(ROOMS)[int(rng.integers(len(ROOMS)))]
        if verb in ('go', 'guide'):
            c.location = self.locations[int(rng.integers(len(self.locations)))]
        if verb in ('object_yesno', 'bring'):
            c.obj = list(OBJECTS)[int(rng.integers(len(OBJECTS)))]
            c.furniture = OBJECT_SURFACES[int(rng.integers(len(OBJECT_SURFACES)))]
        c.text = self.render(c)
        return c

    def render(self, c: Command) -> str:
        rng = self.rng
        t = _expand(rng, TEMPLATES[c.verb][int(rng.integers(len(TEMPLATES[c.verb])))])
        loc = None
        if c.location:
            loc = _pick(rng, ROOMS, c.location) if c.location in ROOMS else _pick(rng, FURNITURE, c.location)
        t = (t.replace('{color}', _pick(rng, COLORS, c.person) if c.person else '')
                .replace('{room}', _pick(rng, ROOMS, c.room) if c.room else '')
                .replace('{loc}', loc or '')
                .replace('{obj}', OBJECTS[c.obj][0 if c.verb == 'object_yesno' else 1] if c.obj else '')
                .replace('{furn}', _pick(rng, FURNITURE, c.furniture) if c.furniture else ''))
        return re.sub(r'\ba el\b', 'al', re.sub(r'\bde el\b', 'del', t))


# ================================================================================================ intérprete
def _alts(d: dict) -> list[tuple[str, str]]:
    """(frase normalizada, id) ordenadas de mayor a menor longitud (la más específica gana)."""
    out = [(_norm(p), k) for k, ps in d.items() for p in ps]
    return sorted(out, key=lambda x: -len(x[0]))


_ROOM_A, _FURN_A, _COLOR_A, _OBJ_A = _alts(ROOMS), _alts(FURNITURE), _alts(COLORS), _alts(OBJECTS)
_VERB_PATTERNS = [   # (verbo, expresión sobre el texto normalizado); el orden resuelve ambigüedades
    ('bring', r'\b(traeme|alcanzame)\b'),
    ('guide', r'\b(guia|lleva|guiala|llevala)\b|\bacompana a .* (hasta|a) '),
    ('follow', r'\b(segui|seguila|acompana)\b'),
    ('count_people', r'\bcuantas personas\b'),
    ('object_yesno', r'\bhay (una|un|la|el)\b'),
    ('where_person', r'\b(en que (habitacion|cuarto) esta|donde esta)\b'),
    ('greet', r'\b(saludala|saluda)\b'),
    ('go', r'\b(anda|ve|dirigite|navega|tenes que ir)\b'),
]


def _find(text: str, alts) -> list[tuple[int, str]]:
    """Ocurrencias (inicio, fin, id) sin solapamiento, prefiriendo la frase más larga."""
    taken = [False] * (len(text) + 1)
    hits = []
    for phrase, key in alts:
        for m in re.finditer(r'(?<![a-z0-9])' + re.escape(phrase) + r'(?![a-z0-9])', text):
            if not any(taken[m.start():m.end()]):
                hits.append((m.start(), m.end(), key))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    return sorted(hits)


class CommandParser:
    """Intérprete determinista de la gramática cerrada: texto -> Command (o None si no se reconoce)."""

    def parse(self, text: str) -> Command | None:
        t = _norm(text)
        verb = next((v for v, pat in _VERB_PATTERNS if re.search(pat, t)), None)
        if verb is None:
            return None
        objs = [(p, k) for p, _, k in _find(t, _OBJ_A)]
        # los nombres de objeto contienen palabras de color ("naranja"): las personas solo tras "camiseta"
        colors = [(p, k) for p, _, k in _find(t, _COLOR_A) if t[max(0, p - 9):p].strip().endswith('camiseta')]
        fspans = _find(t, _FURN_A)
        furn = [(p, k) for p, _, k in fspans]
        # "la cocina electrica" / "la mesa del lavadero" son muebles: la habitación no puede caer dentro de su frase
        rooms = [(p, k) for p, e, k in _find(t, _ROOM_A) if not any(fp <= p < fe for fp, fe, _ in fspans)]
        c = Command(verb=verb, text=text, requires_grasp=verb in NEEDS_GRASP)
        if verb in ('greet', 'where_person', 'follow', 'guide'):
            if not colors:
                return None
            c.person = colors[0][1]
        if verb in ('greet', 'count_people', 'follow', 'guide'):
            if not rooms:
                return None
            c.room = rooms[0][1]
        if verb == 'go':
            locs = sorted(rooms + furn)
            if len(locs) != 1:
                return None
            c.location = locs[0][1]
        if verb == 'guide':
            # destino: la ubicación mencionada después de la habitación de origen
            after = sorted([(p, k) for p, k in rooms + furn if p > rooms[0][0]])
            if not after:
                return None
            c.location = after[-1][1]
        if verb in ('object_yesno', 'bring'):
            if not objs or not furn:
                return None
            c.obj, c.furniture = objs[0][1], furn[0][1]
        return c
