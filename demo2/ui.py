"""Python-only Jupyter UI; callbacks perform simulation and local profile I/O."""
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from .profiles import PROFILE_DIR, save_profile, load_profile, list_profiles
from .simulation import load_sources, simulate
from .presentation import show_result


def build_controls(profile_directory=PROFILE_DIR, output_directory=None):
    output_directory = Path(output_directory or Path(__file__).parent/"local"/"latest")
    name = widgets.Text(value="default", description="Profile name:")
    engine = widgets.Checkbox(value=True, description="Reduce engine")
    keyboard = widgets.Checkbox(value=False, description="Reduce keyboard (experimental)", indent=False)
    preserve = widgets.Checkbox(value=True, description="Preserve speech", disabled=True)
    saved = widgets.Dropdown(options=["— saved profiles —"]+list_profiles(profile_directory), description="Saved:")
    load_button = widgets.Button(description="Load profile")
    save_button = widgets.Button(description="Save preferences", button_style="info")
    run_button = widgets.Button(description="Run cancellation", button_style="success")
    status = widgets.Label(value="Select sounds, then run. Preferences are saved only when you click Save.")
    output = widgets.Output()
    state = {"sources":None, "last_result":None}

    def selection():
        return [key for key, control in [("engine",engine),("keyboard",keyboard)] if control.value]

    def apply(profile):
        engine.value = "engine" in profile["reduce"]
        keyboard.value = "keyboard" in profile["reduce"]

    def save(_):
        try:
            save_profile(name.value, selection(), profile_directory)
            saved.options = ["— saved profiles —"]+list_profiles(profile_directory)
            saved.value = name.value
            status.value = f"Saved {name.value}. Reopening and loading this profile restores these choices."
        except (ValueError, OSError) as error:
            status.value = str(error)

    def load(_):
        try:
            apply(load_profile(name.value, profile_directory))
            status.value = f"Loaded {name.value}. Run cancellation to hear these settings."
        except (ValueError, OSError) as error:
            status.value = str(error)

    def choose(change):
        if change["new"] != "— saved profiles —":
            name.value = change["new"]

    def run(_):
        run_button.disabled = True
        status.value = "Running simulation; first run may download example recordings…"
        with output:
            output.clear_output(wait=True)
            try:
                if state["sources"] is None:
                    state["sources"], _ = load_sources()
                result = simulate(state["sources"], selection())
                state["last_result"] = result
                show_result(result, output_directory)
                status.value = "Finished. Compare playback and metrics; negative reduction means amplification."
            except Exception as error:
                status.value = f"Could not run: {error}"
                print(status.value)
            finally:
                run_button.disabled = False

    saved.observe(choose, names="value")
    save_button.on_click(save); load_button.on_click(load); run_button.on_click(run)
    if "default" in list_profiles(profile_directory):
        load(None)
    box = widgets.VBox([
        widgets.HTML("<h3>Choose what to reduce</h3><p>Speech + engine + keyboard, with separately known source references. "
                     "Offline ear-signal simulation; not live microphone separation or hardware ANC.</p>"),
        widgets.HBox([engine, keyboard]), preserve,
        widgets.HBox([name, saved]), widgets.HBox([save_button, load_button, run_button]), status, output])
    # Expose controls for callback-level regression tests and notebook customization.
    controls = {"name":name,"engine":engine,"keyboard":keyboard,"save":save_button,
                "load":load_button,"run":run_button,"status":status,"state":state}
    return box, controls


def show_controls():
    box, controls = build_controls()
    display(box)
    return controls
