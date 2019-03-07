gnome-terminal -e 'bash -c roscore'
sleep 2
gnome-terminal -e 'roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=firefly world_name:=basic'
sleep 2
gnome-terminal -e 'python Viz_Dron.py'
gnome-terminal -e 'python Moves.py'
gnome-terminal -e 'bash -c "source /opt/venvs/drone-venv/bin/activate;python Mov_Ev.py"'
