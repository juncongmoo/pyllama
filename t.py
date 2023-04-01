import subprocess

# Define the command to be executed
cmd = "echo 'Hello, World!' && sleep 1 && echo 'Goodbye, World!'"

# Create the subprocess and redirect output to the current terminal
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Loop over the subprocess output and print it immediately
while True:
    output = proc.stdout.readline().decode().rstrip()
    if output == '' and proc.poll() is not None:
        break
    print(output)
    
# Wait for the subprocess to complete and get the return code
return_code = proc.wait()

# Print the return code to the current terminal
print(f"Subprocess returned with code {return_code}")


