const fs = require("fs");

// Function to clean track name by removing special characters and control codes
function cleanTrackName(trackName) {
  return trackName
    .replace(/[\u0000-\u001F\u007F-\u009F\u2028\u2029]/g, "")
    .replace(/[\\\/]+/g, "") // Remove backslashes and forward slashes
    .replace(/\s+/g, " ")
    .trim();
}

// Function to clean and truncate file path (not used in the final output)
function cleanAndTruncateFilePath(filePath) {
  const cleanedPath = filePath.replace(/[\u0000-\u001F\u007F-\u009F\u2028\u2029]/g, "").replace(/\s+/g, " ").trim();
  return cleanedPath.length > 30 ? cleanedPath.substring(0, 30) : cleanedPath;
}

function parseSeratoData(filePath) {
  const tracks = [];

  try {
    console.log(`Reading file from: ${filePath}`);
    const content = fs.readFileSync(filePath, "utf-8");
    console.log("File read successfully");

    const trackEntries = content.split("tsng");

    for (let i = 1; i < trackEntries.length; i++) {
      try {
        const entry = trackEntries[i];

        // Extract the track name
        const trackNameStart = 0; // Start from the beginning of the entry
        const trackNameEnd = entry.indexOf("tgen");
        const trackName = entry
          .substring(trackNameStart, trackNameEnd)
          .replace(/\0/g, "")
          .trim();
        const cleanedTrackName = cleanTrackName(trackName);

        // Extract the key
        const keyStart = entry.indexOf("tkey") + 4;
        const keyEnd = keyStart + 10;
        const rawKey = entry
          .substring(keyStart, keyEnd)
          .replace(/\0/g, "")
          .trim();
        const keyMatch = rawKey.match(/[A-G][#b]?[mM]?/);
        const key = keyMatch ? keyMatch[0] : null;

        // Extract the BPM
        const bpmStart = entry.indexOf("tbpm") + 4;
        const bpmEnd = entry.indexOf("tcom");
        const bpmStr = entry
          .substring(bpmStart, bpmEnd)
          .replace(/\0/g, "")
          .trim();
        const bpm = bpmStr ? parseFloat(bpmStr) : null;

        // Extract the file path (not added to the final output)
        const filePathStart = entry.indexOf("pfil") + 4;
        const filePathEnd = entry.indexOf("tsng", filePathStart);
        const filePath = entry
          .substring(filePathStart, filePathEnd)
          .replace(/\0/g, "")
          .trim();
        const cleanedAndTruncatedFilePath = cleanAndTruncateFilePath(filePath);

        // Create a track object with only track, key, and bpm
        const track = {
          ...(cleanedTrackName && { track: cleanedTrackName }),
          ...(key && { key: key }),
          ...(bpm && !isNaN(bpm) && { bpm: bpm }),
        };

        // Only add the track to the list if all three fields are present
        if (track.track && track.key && track.bpm !== undefined) {
          tracks.push(track);
        }
      } catch (error) {
        console.error(`Error parsing entry: ${error}`);
        continue;
      }
    }
  } catch (error) {
    console.error(`Error reading file: ${error}`);
    return [];
  }

  return tracks;
}

const seratoDataFile = "/Users/ben/Music/_Serato_/database V2";

if (!fs.existsSync(seratoDataFile)) {
  console.error(`File not found: ${seratoDataFile}`);
} else {
  const tracks = parseSeratoData(seratoDataFile);

  if (tracks.length > 0) {
    fs.writeFileSync("tracks.json", JSON.stringify(tracks, null, 4));
    console.log(`Extracted ${tracks.length} tracks and saved to tracks.json`);
  } else {
    console.log("No tracks were extracted.");
  }
}
