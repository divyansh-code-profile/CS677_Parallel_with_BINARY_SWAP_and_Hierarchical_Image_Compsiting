import subprocess

# Define the path to the commands file
commands_file = '/mnt/data/commands.txt'

# Open the commands file and read the commands
with open(commands_file, 'r') as file:
    lines = file.readlines()

# Filter out empty lines and strip whitespace
commands = [line.strip() for line in lines if line.strip()]

# Loop over each command, execute it, and capture the output
for i, command in enumerate(commands):
    print(f"Executing command {i+1}: {command}")
    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        # Display the output
        print(f"Output of command {i+1}:\n{result.stdout}\n")
    except subprocess.CalledProcessError as e:
        # Print error if command fails
        print(f"Error executing command {i+1}:\n{e.stderr}\n")
