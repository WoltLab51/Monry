import React, { useEffect, useState } from "react";
import { View, Text, ScrollView, TextInput, Button, StyleSheet } from "react-native";
import { fetchCompleteData } from "./api/fmp";
import { analyzeFactors } from "./core/factorEngine";
import { smartScoreStock } from "./core/smartScoringModel";
import { saveTrade, getTradeHistory } from "./core/tradeMemory";
import { callGptAssistant } from "./core/gptAssistant";

const WATCHLIST = ["AAPL", "MSFT", "TSLA"];

export default function App() {
  const [log, setLog] = useState([]);
  const [gptInput, setGptInput] = useState("");
  const [gptOutput, setGptOutput] = useState("");
  const [history, setHistory] = useState([]);

  useEffect(() => {
    runAutoAnalysis();
    setHistory(getTradeHistory());
  }, []);

  const runAutoAnalysis = async () => {
    setLog((prev) => [...prev, "🧠 MONRY startet Analyse..."]);
    for (const symbol of WATCHLIST) {
      const rawData = await fetchCompleteData(symbol);
      const factors = await analyzeFactors(rawData);
      const result = smartScoreStock(factors);
      saveTrade({ symbol, date: rawData.date, ...factors, ...result });
      setLog((prev) => [...prev, `✅ ${symbol}: ${result.recommendation} (Score ${result.score})`]);
    }
    setHistory(getTradeHistory());
  };

  const handleGptSend = async () => {
    const response = await callGptAssistant(gptInput);
    setGptOutput(response);
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.header}>📊 MONRY Dashboard</Text>
      {log.map((entry, i) => (
        <Text key={i}>• {entry}</Text>
      ))}
      <View style={styles.section}>
        <Text style={styles.subHeader}>🧠 GPT-Assistent:</Text>
        <TextInput
          value={gptInput}
          onChangeText={setGptInput}
          placeholder="Frage z.B. Warum wurde AAPL empfohlen?"
          style={styles.input}
        />
        <Button title="Frage senden" onPress={handleGptSend} />
        <Text style={styles.response}>{gptOutput}</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, marginTop: 40 },
  header: { fontSize: 24, fontWeight: "bold", marginBottom: 20 },
  subHeader: { fontSize: 16, fontWeight: "bold", marginVertical: 10 },
  section: { marginTop: 20 },
  input: { borderWidth: 1, padding: 10, marginVertical: 10 },
  response: { marginTop: 10, fontStyle: "italic" },
});