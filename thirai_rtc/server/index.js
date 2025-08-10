const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(express.static(path.join(__dirname, '../public')));

const rooms = new Map();

io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  socket.on('join-room', (data) => {
    const { roomId, userName } = data;
    socket.join(roomId);
    socket.userName = userName;
    socket.roomId = roomId;

    if (!rooms.has(roomId)) {
      rooms.set(roomId, new Set());
    }
    rooms.get(roomId).add(socket.id);

    socket.to(roomId).emit('user-joined', {
      socketId: socket.id,
      userName: userName
    });

    const roomUsers = Array.from(rooms.get(roomId))
      .map(id => {
        const userSocket = io.sockets.sockets.get(id);
        return userSocket ? { socketId: id, userName: userSocket.userName } : null;
      })
      .filter(user => user !== null);

    socket.emit('room-users', roomUsers.filter(user => user.socketId !== socket.id));

    console.log(`${userName} joined room ${roomId}`);
  });

  socket.on('offer', (data) => {
    socket.to(data.target).emit('offer', {
      offer: data.offer,
      sender: socket.id
    });
  });

  socket.on('answer', (data) => {
    socket.to(data.target).emit('answer', {
      answer: data.answer,
      sender: socket.id
    });
  });

  socket.on('ice-candidate', (data) => {
    socket.to(data.target).emit('ice-candidate', {
      candidate: data.candidate,
      sender: socket.id
    });
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
    
    if (socket.roomId && rooms.has(socket.roomId)) {
      rooms.get(socket.roomId).delete(socket.id);
      if (rooms.get(socket.roomId).size === 0) {
        rooms.delete(socket.roomId);
      } else {
        socket.to(socket.roomId).emit('user-left', socket.id);
      }
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});