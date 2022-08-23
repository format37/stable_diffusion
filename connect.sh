# get id of container with name kandinsky_generator_1 and store to variable id
id=$(sudo docker ps -a | grep generator_stable_diffusion_1 | awk '{print $1}')
# connect bash to container with id $id
sudo docker exec -it $id /bin/bash