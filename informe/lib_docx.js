// Utilidades compartidas para construir el informe .docx (estilo paper academico).
const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, ShadingType, HeadingLevel, ImageRun,
} = require("docx");

const CONTENT_WIDTH = 9360; // US Letter, margenes 1"

// --- Lectura de CSV (simple, sin comillas anidadas complejas) ---
function parseCSV(texto) {
  const filas = [];
  let campo = "", fila = [], enComillas = false;
  for (let i = 0; i < texto.length; i++) {
    const c = texto[i];
    if (enComillas) {
      if (c === '"' && texto[i + 1] === '"') { campo += '"'; i++; }
      else if (c === '"') enComillas = false;
      else campo += c;
    } else if (c === '"') enComillas = true;
    else if (c === ",") { fila.push(campo); campo = ""; }
    else if (c === "\n") { fila.push(campo); filas.push(fila); fila = []; campo = ""; }
    else if (c === "\r") { /* skip */ }
    else campo += c;
  }
  if (campo.length || fila.length) { fila.push(campo); filas.push(fila); }
  return filas.filter(f => f.length > 1 || (f.length === 1 && f[0].trim() !== ""));
}

function leerCSV(ruta) {
  let texto = fs.readFileSync(ruta, "utf8");
  if (texto.charCodeAt(0) === 0xFEFF) texto = texto.slice(1); // BOM
  const filas = parseCSV(texto);
  const cabecera = filas[0].map(h => h.replace(/^﻿/, "").trim());
  return filas.slice(1).map(f => Object.fromEntries(cabecera.map((h, i) => [h, (f[i] ?? "").trim()])));
}

function existe(ruta) { return fs.existsSync(ruta); }

// --- Constructores de contenido ---
function titulo(texto) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [new TextRun({ text: texto, bold: true, size: 32, font: "Times New Roman" })],
  });
}

function subtitulo(texto) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text: texto, italics: true, size: 24, font: "Times New Roman" })],
  });
}

function h1(texto) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 130 },
    children: [new TextRun({ text: texto, bold: true, size: 26, font: "Times New Roman" })] });
}

function h2(texto) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 100 },
    children: [new TextRun({ text: texto, bold: true, size: 24, font: "Times New Roman" })] });
}

function p(texto, opts = {}) {
  return new Paragraph({
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 120, line: 276 },
    children: [new TextRun({ text: texto, size: 22, font: "Times New Roman",
                             italics: !!opts.italics, bold: !!opts.bold })],
  });
}

function pRuns(runs, opts = {}) {
  return new Paragraph({
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 120, line: 276 },
    children: runs.map(r => new TextRun({ text: r.t, bold: !!r.b, italics: !!r.i,
                                          size: 22, font: "Times New Roman" })),
  });
}

function bullet(texto) {
  return new Paragraph({ numbering: { reference: "vinetas", level: 0 },
    spacing: { after: 60, line: 264 },
    children: [new TextRun({ text: texto, size: 22, font: "Times New Roman" })] });
}

function celda(texto, { header = false, ancho, alinear = AlignmentType.LEFT } = {}) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  return new TableCell({
    borders: { top: border, bottom: border, left: border, right: border },
    width: { size: ancho, type: WidthType.DXA },
    shading: header ? { fill: "2E5496", type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: alinear, spacing: { after: 0 },
      children: [new TextRun({ text: texto, bold: header, color: header ? "FFFFFF" : "000000",
                               size: 18, font: "Times New Roman" })] })],
  });
}

// filas: array de arrays de strings. anchos: array que suma CONTENT_WIDTH.
// alineaciones: opcional, por columna.
function tabla(cabeceras, filas, anchos, alineaciones) {
  const al = alineaciones || cabeceras.map(() => AlignmentType.LEFT);
  const rows = [
    new TableRow({ tableHeader: true, children: cabeceras.map((c, i) =>
      celda(c, { header: true, ancho: anchos[i], alinear: AlignmentType.CENTER })) }),
    ...filas.map(f => new TableRow({ children: f.map((v, i) =>
      celda(String(v), { ancho: anchos[i], alinear: al[i] })) })),
  ];
  return new Table({ width: { size: CONTENT_WIDTH, type: WidthType.DXA },
                     columnWidths: anchos, rows });
}

function pieTabla(texto) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 180 },
    children: [new TextRun({ text: texto, italics: true, size: 18, font: "Times New Roman" })] });
}

function imagen(ruta, { ancho = 460, alto = 300, tipo = "png" } = {}) {
  if (!existe(ruta)) return p(`[figura no disponible: ${path.basename(ruta)}]`, { italics: true, center: true });
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new ImageRun({ type: tipo, data: fs.readFileSync(ruta),
      transformation: { width: ancho, height: alto },
      altText: { title: "figura", description: path.basename(ruta), name: path.basename(ruta) } })] });
}

const CLASES = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"];

module.exports = {
  CONTENT_WIDTH, CLASES, leerCSV, existe, titulo, subtitulo, h1, h2, p, pRuns,
  bullet, tabla, pieTabla, imagen,
};
