# Let's test the state dictionary logic
state = {
    "cutoff_num": 480,
    "cutoff_slide": 480,
    "lgd_num": 3.73,
    "lgd_slide": 3.73,
    "margin_num": 0.15,
    "margin_slide": 0.15
}

def on_cutoff_num_change():
    state["cutoff_slide"] = int(state["cutoff_num"])

def on_cutoff_slide_change():
    state["cutoff_num"] = int(state["cutoff_slide"])

# Test 1: user changes number input to 520
state["cutoff_num"] = 520
on_cutoff_num_change()
assert state["cutoff_slide"] == 520, f"Failed: slider is {state['cutoff_slide']}"

# Test 2: user drags slider to 450
state["cutoff_slide"] = 450
on_cutoff_slide_change()
assert state["cutoff_num"] == 450, f"Failed: num is {state['cutoff_num']}"

print("All synchronization assertions passed successfully!")
