
import dotenv from "dotenv";
import app from "./app.js";
import { connectDB } from "./dB/db.connect.js";




dotenv.config({
});

const PORT = process.env.PORT || 1000;





connectDB()
  .then(() => {
    app.listen(PORT, () => {
      console.log(` HTTPS Server running on http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("Failed to connect to DB:", err);
  });
