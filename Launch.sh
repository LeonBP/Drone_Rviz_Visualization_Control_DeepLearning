gnome-terminal -e 'bash -c roscore'
sleep 2
gnome-terminal -e 'bash -c rqt'
gnome-terminal -e 'python Moves.py'
gnome-terminal -e 'python Viz_Dron.py'
gnome-terminal -e 'python Mov_Ev.py'
