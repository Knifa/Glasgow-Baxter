class State(object): 
    def __init__(self, node):
        self._node = node

    def enter(self):
        pass

    def run_step(self):
        pass

    def next(self):
        pass

    def exit(self):
        pass

    def on_left_image_received(self, img):
        pass

    def on_right_image_received(self, img):
        pass

####################################################################################################

class StateMachine(object):
    def __init__(self, node):
        self._node = node

    ############################################################################

    def start(self):
        self._current_state.enter()

    def run_step(self):
        if self._current_state is None:
            return

        self._current_state.run_step()

        next_state = self._current_state.next()
        if not next_state is None and not self._current_state is next_state:
            self._current_state.exit()
            next_state.enter()
            self._current_state = next_state

    ############################################################################

    def on_left_image_received(self, img):
        if not self._current_state is None:
            self._current_state.on_left_image_received(img)

    def on_right_image_received(self, img):
        if not self._current_state is None:
            self._current_state.on_right_image_received(img)