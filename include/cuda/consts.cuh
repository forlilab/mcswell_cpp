#pragma once
#include <stdint.h>

static constexpr int WATER_SIZE = 12;
static constexpr int W_STRIDE = 4;
static constexpr int ATOMS_PER_W = 3;
static constexpr int ATOM_FEATURES = 7;

// compact water layout indices
static constexpr int OX    = 0;
static constexpr int OY    = 1;
static constexpr int OZ    = 2;
static constexpr int ORES  = 3;

static constexpr int H1X   = 4;
static constexpr int H1Y   = 5;
static constexpr int H1Z   = 6;
static constexpr int H1RES = 7;

static constexpr int H2X   = 8;
static constexpr int H2Y   = 9;
static constexpr int H2Z   = 10;
static constexpr int H2RES = 11;

static constexpr float KT = 0.59616123; // BOLTZMANN_K * TEMPERATURE
static constexpr float BETA = 1.0 / KT;
static constexpr float STANDARD_VOLUME = 29.914f;
static constexpr float BOLTZMANN_K = 0.0019872041; // Boltzmann constant (kcal/mol)
static constexpr float TEMPERATURE = 300.; // Temperature used for Boltzmann sampling (K)
static constexpr float BULK_WATER_DENSITY = 0.0334;
static constexpr float EPSILON_RF = 80.0;
static constexpr float RF_CUTOFF = 12.0;