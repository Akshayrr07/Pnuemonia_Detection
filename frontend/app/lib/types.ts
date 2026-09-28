/**
 * Shared prediction response shape.
 * Mirrors src/inference/schemas.HierarchicalPrediction.to_dict().
 */
export interface PredictResponse {
  primary_prediction: "Normal" | "Pneumonia";
  primary_confidence: number;
  subtype_prediction: string | null;
  subtype_confidence: number | null;
  probabilities: Record<string, number>;
  model_outputs: {
    binary: Record<string, unknown>;
    subtype: Record<string, unknown>;
  };
  disclaimer: string;
  heatmap_b64: string | null;
}

export type PrimaryLabel = "Normal" | "Pneumonia";

export const DISEASE_LABELS: readonly PrimaryLabel[] = [
  "Normal",
  "Pneumonia",
] as const;

export const SUBTYPE_LABELS = [
  "Bacterial Pneumonia",
  "Viral Pneumonia",
  "Uncertain Pneumonia Subtype",
] as const;

export const PROBABILITY_LABELS = [
  "Normal",
  "Pneumonia",
  "Bacterial Pneumonia",
  "Viral Pneumonia",
] as const;
