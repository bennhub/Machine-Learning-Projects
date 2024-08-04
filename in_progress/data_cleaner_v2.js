const fs = require("fs");

// Function to clean track name by removing special characters and control codes
function cleanTrackName(trackName) {
  return trackName
    .replace(/[\u0000-\u001F\u007F-\u009F\u2028\u2029]/g, "")
    .replace(/[\\\/]+/g, "") // Remove backslashes and forward slashes
    .replace(/\s+/g, " ")
    .trim();
}

// Function to handle the processing of the JSON data
function processJsonData(data) {
  return data.map(item => {
    // Extract and clean fields
    const cleanedTrackName = cleanTrackName(item.Title || "unknown");
    const key = item.Key || null;
    const bpm = item.BPM || null;
    const artist = item.Artist || "unknown";

    // Create a track object with only required fields
    return {
      artist: artist,
      track: cleanedTrackName,
      bpm: bpm,
      key: key
    };
  }).filter(track => track.track && track.key && track.bpm !== undefined);
}

const seratoDataFile = "DjayPro2.json";

if (!fs.existsSync(seratoDataFile)) {
  console.error(`File not found: ${seratoDataFile}`);
} else {
  try {
    console.log(`Reading file from: ${seratoDataFile}`);
    const content = fs.readFileSync(seratoDataFile, "utf-8");
    console.log("File read successfully");

    // Parse JSON content
    const data = JSON.parse(content);

    // Process JSON data
    const tracks = processJsonData(data);

    if (tracks.length > 0) {
      fs.writeFileSync("json-files/extractV2List.json", JSON.stringify(tracks, null, 4));
      console.log(`Extracted ${tracks.length} tracks and saved to extractV2List.json`);
    } else {
      console.log("No tracks were extracted.");
    }
  } catch (error) {
    console.error(`Error reading or parsing file: ${error}`);
  }
}
