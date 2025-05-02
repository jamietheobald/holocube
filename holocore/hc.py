# instantiate the classes needed for 5 sided holocore operation and
# set up the namespace for import by run and experiments

import holocore.control as control
import holocore.windows as windows
import holocore.schedulers as schedulers
import holocore.stimuli as stim
import holocore.arduino as ard
import holocore.tools as tools

# objects we need in run and exps
control = control.Control_Window()
window = windows.Holocube_Window()
scheduler = schedulers.Scheduler()
arduino = ard.Arduino()
