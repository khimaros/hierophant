"""configs/<org>--<repo>.toml and the machine's own config.toml: what a command line can leave out.

a model's config can be shared, so it is data about the model and its search and nothing
else: it can name no path, url or program. those only come from flags or the machine's file."""

import json
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
GGUF_SUFFIX = re.compile(r"-gguf$", re.IGNORECASE)  # a quant repo's name: <Model>-GGUF
RECIPE = re.compile(r"L(?P<index>\d+(?:\.\d+)?)(?:-pcs(?P<pcs>\d+))?(?P<residual>-res)?$")
DIGITS = 4  # of a score in a config
WEIGHT_MATCH = 1e-6  # a config's weight is the sidecar's, printed with %g
# [search] keys a model's config may set, which are hierophant build's flags of those names
SEARCH_KEYS = ("beam", "first_pcs", "first_prompts", "indices", "max_candidates", "max_kld", "report_prompts",
               "think_budget", "think_effort", "think_prompts")
# [defaults] keys of the machine's file, and the flag each one sets
MACHINE_FLAGS = {"ngl": "-ngl", "parallel": "-np", "judge": "--judge", "progress": "--progress"}
PUBLISH_DEFAULTS = ("user", "upload")  # [defaults] keys only publish reads


class ConfigError(Exception):
    """a config that cannot be used, or a MODEL that names none."""


def configs_dir():
    return Path(os.environ.get("HIEROPHANT_CONFIGS") or CONFIGS)


def load_configs(directory=None):
    """{file stem: config} of every model config."""
    return {p.stem: tomllib.loads(p.read_text()) for p in sorted(Path(directory or configs_dir()).glob("*.toml"))}


def machine_path():
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "hierophant" / "config.toml"


def load_machine(path=None):
    """the machine's file, or nothing where it has none."""
    path = Path(path or machine_path())
    return tomllib.loads(path.read_text()) if path.exists() else {}


def find_config(name, configs):
    """the stem of the one config whose file name holds name, case aside."""
    found = [stem for stem in configs if name.lower() in stem.lower()]
    if len(found) != 1:
        raise ConfigError(f"{name} matches {', '.join(found)}" if found else
                          f"no config matches {name}: known are {', '.join(configs) or 'none'}")
    return found[0]


def config_of_repo(repo, configs):
    """the stem of the config of a hub repo, or None."""
    return next((stem for stem, c in configs.items() if c["model"]["repo"] == repo), None)


def quant_of(stem, config, quant):
    """the quant to use: the one asked for, else the config's only one."""
    known = list(config.get("quant", {}))
    if quant:
        return quant
    if len(known) != 1:
        raise ConfigError(f"{stem} needs --quant: it knows {', '.join(known) or 'no quant yet'}")
    return known[0]


def resolve(model, quant, configs):
    """(flags naming the base, its config or None) for a command line's MODEL: a file, a
    hub repo, or any unique part of a config's name."""
    if model.endswith(".gguf") or Path(model).exists():
        return ["-m", model], None
    if "/" not in model:
        stem = find_config(model, configs)
        config = configs[stem]
        return ["--hf-repo", f"{config['model']['repo']}:{quant_of(stem, config, quant)}"], config
    repo, _, tag = model.partition(":")
    stem = config_of_repo(repo, configs)
    tag = quant or tag
    if stem:
        return ["--hf-repo", f"{repo}:{quant_of(stem, configs[stem], tag)}"], configs[stem]
    return ["--hf-repo", f"{repo}:{tag}" if tag else repo], None


def flag(name, value):
    return [name, str(value)]


def machine_flags(machine):
    """the flags of the machine's [defaults]."""
    defaults = machine.get("defaults", {})
    unknown = sorted(set(defaults) - {*MACHINE_FLAGS, *PUBLISH_DEFAULTS})
    if unknown:
        raise ConfigError(f"unknown [defaults] keys: {', '.join(unknown)}")
    out = []
    for k, name in MACHINE_FLAGS.items():
        if k in defaults and defaults[k] is not False:
            out += [name] if defaults[k] is True else flag(name, defaults[k])
    return out


def publish_flags(machine):
    """the flags of the [defaults] only publish reads. upload must be exactly true, since it
    makes a plain publish public."""
    defaults = machine.get("defaults", {})
    user = defaults.get("user")
    return [*(["--user", user] if user else []), *(["--upload"] if defaults.get("upload") is True else [])]


def search_flags(config):
    """the flags of a model's [search]."""
    search = (config or {}).get("search", {})
    unknown = sorted(set(search) - set(SEARCH_KEYS))
    if unknown:
        raise ConfigError(f"unknown [search] keys: {', '.join(unknown)}")
    return [x for k, v in search.items() for x in flag("--" + k.replace("_", "-"), v)]


def recipe_name(r):
    """a recipe's name, as hierophant_auto.key names a candidate: L33-pcs8-res."""
    return f"L{r['index']:g}" + (f"-pcs{r['pcs']}" if r.get("pcs") else "") + ("-res" if r.get("residual") else "")


def built_recipe(directions):
    """the name of the recipe a sidecar's directions describe."""
    d = directions
    return recipe_name({"index": d["index"], "pcs": d.get("harmless_pcs"), "residual": "residual" in d})


def parse_recipe(name):
    """{index, pcs, residual} of a recipe's name."""
    m = RECIPE.match(name)
    if not m:
        raise ConfigError(f"not a recipe: {name}: try L33, L33-pcs8 or L33-pcs8-res")
    return {"index": float(m["index"]), "pcs": int(m["pcs"] or 0), "residual": bool(m["residual"])}


def accepted_quant(config, quant):
    """the table of the recipe to follow on a quant: its own, else any other quant's."""
    quants = (config or {}).get("quant", {})
    return next((q for q in (quants.get(quant, {}), *quants.values()) if "recipe" in q), None)


def accepted_recipe(config, quant):
    """the name of the recipe to follow on a quant, or None."""
    q = accepted_quant(config, quant)
    return recipe_name(q["recipe"]) if q else None


def summary(configs):
    """one line per config and quant: its recipe and its scores."""
    lines = []
    for stem, c in configs.items():
        for quant, q in c.get("quant", {}).items():
            r, e = q.get("recipe"), q.get("eval", {})
            scores = f"refusals {e['refusals']:.3f} on {e['n']}" if e else "not scored"
            lines.append(f"{stem} {quant}: {recipe_name(r) if r else 'no recipe'}, {scores}")
    return "\n".join(lines)


def run_dir(repo, quant):
    """a run's default directory, which hierophant_run.default_out also names."""
    return Path("out") / f"{repo.replace('/', '--')}--{quant}"


def stem_of(repo):
    return repo.replace("/", "--")


def recipe_args(config, quant):
    """the flags that follow a config's accepted recipe on a quant, or none."""
    q = accepted_quant(config, quant)
    return ["--recipe", recipe_name(q["recipe"]), "--recipe-weight", str(q["weight"])] if q else []


def check_accepted(best, table):
    """that a run's best is what its config accepted: the same recipe at the same weight."""
    side = json.loads(best.with_suffix(".json").read_text())
    built = built_recipe(side["directions"])
    if built != recipe_name(table["recipe"]):
        raise ConfigError(f"{best} is {built}, not the accepted {recipe_name(table['recipe'])}")
    weight = side["curve"]["max_weight"]
    if abs(weight - table.get("weight", weight)) > WEIGHT_MATCH:
        raise ConfigError(f"{best} has weight {weight:g}, not the accepted {table['weight']:g}: "
                          "if another run built it, name it with --name")


def publish_args(stem, config, machine, quant="", name=""):
    """hierophant_publish's arguments for a config: each accepted quant's best, from run
    name if given, which must still be what the config accepted, and the model's license."""
    if not config["model"].get("license"):
        raise ConfigError(f"{stem} has no license: set the one its model card names in its config")
    adapters = []
    for q, table in config.get("quant", {}).items():
        if "recipe" not in table or quant not in ("", q):
            continue
        root = run_dir(config["model"]["repo"], q)
        best = (root / "runs" / name if name else root) / "best.gguf"
        if not best.exists():
            raise ConfigError(f"no {best}: run hierophant build {stem} --quant {q}")
        check_accepted(best, table)
        adapters.append(str(best))
    if not adapters:
        raise ConfigError(f"{stem} has no accepted recipe to publish")
    return [*adapters, "--license", config["model"]["license"], *publish_flags(machine)]


def run_best(repo, quant, machine):
    """hierophant_publish's arguments for the best of a run with no config."""
    best = run_dir(repo, quant) / "best.gguf"
    if not best.exists():
        raise ConfigError(f"no {best}: run hierophant build {repo}:{quant}")
    return [str(best), *publish_flags(machine)]


def new_config(repo):
    """a config knowing only its repo."""
    name = GGUF_SUFFIX.sub("", repo.rpartition("/")[2])
    return f'[model]\nname = "{name}"\nrepo = "{repo}"\nlicense = "" # unconfirmed; publish needs it\n'


def quant_section(quant, side, think=None):
    """a quant's tables, from its adapter's sidecar and any short thinking scores."""
    d, e = side["directions"], side["eval"]
    residual = "true" if "residual" in d else "false"
    recipe = f"{{ index = {d['index']:g}, pcs = {d.get('harmless_pcs', 0)}, residual = {residual} }}"
    lines = [f"[quant.{quant}]", f'file = "{side["base"]["file"]}"', f'sha256 = "{side["base"]["sha256"]}"',
             f"recipe = {recipe}", f"weight = {side['curve']['max_weight']:g}", f"scale = {side['scale']:g}", "",
             f"[quant.{quant}.eval]", f"refusals = {round(e['refusals']['adapted'], DIGITS)}",
             f"base = {round(e['refusals']['base'], DIGITS)}", f"n = {e['refusals']['n']}",
             f"kl_first_token = {round(e['kld']['mean'], DIGITS)}"]
    if "kld_seq" in e:
        lines.append(f"kl_text = {round(e['kld_seq']['mean'], DIGITS)}")
    if think:
        lines += [f"think = {round(think['think'], DIGITS)}", f"think_base = {round(think['think_base'], DIGITS)}",
                  f"think_n = {think['think_n']}"]
    final = e.get("final")
    if final:
        lines += [f"loops = {round(final['loops'], DIGITS)}", f"full_refusals = {round(final['markers'], DIGITS)}",
                  f"full_n = {final['n']}"]
    lines.append(f'llama_cpp = "{side["llama_cpp"]}"')
    return "\n".join(lines) + "\n"


def with_quant(text, quant, section):
    """a config's text with one quant's tables replaced, and everything else left as it
    was: tomllib cannot write, and a person's comments and order must survive. comments
    right above a replaced table go with it, since they describe the old numbers."""
    table = rf"(?:^#[^\n]*\n)*^\[quant\.{re.escape(quant)}(?:\.eval)?\]\n(?:(?![\[#])[^\n]*\n)*"
    kept = re.sub(table, "", text, flags=re.MULTILINE).rstrip("\n")
    return f"{kept}\n\n{section}"


def init(repo, directory=None):
    """a config for a repo, unless it has one."""
    path = Path(directory or configs_dir()) / f"{stem_of(repo)}.toml"
    if path.exists():
        raise ConfigError(f"{path} exists")
    path.write_text(new_config(repo))
    return path


def accept(repo, quant, side, think=None, directory=None):
    """write a sidecar's recipe and scores into the repo's config, creating it if needed."""
    path = Path(directory or configs_dir()) / f"{stem_of(repo)}.toml"
    text = path.read_text() if path.exists() else new_config(repo)
    path.write_text(with_quant(text, quant, quant_section(quant, side, think)))
    return path
