import socket

# Server configuration
HOST = '192.168.201.100'  # Localhost
PORT = 6601        # Port to listen on

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind((HOST, PORT))
server_socket.listen()

print(f"Server started. Listening on {HOST}:{PORT}")

# Accept connection from client
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

# Messaging loop
try:
    while True:
        # Receive message from client
        data = conn.recv(1024)
        if not data:
            break
        print(f"Client: {data.decode()}")

        # Reply to client
        message = input("Enter message to client: ")
        #message = "found"
        conn.sendall(message.encode())  # Encode the string message to bytes
finally:
    # Close the connection
    conn.close()
    server_socket.close()
