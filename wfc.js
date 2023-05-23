/**
 *
 * @param {number[]} array
 * @param {float} r
 */
function randomIndice (array, r) {
    let sum = 0;
    let x = 0;
    let i = 0;

    for (; i < array.length; i++) {
        sum += array[i];
    }

    i = 0;
    r *= sum;

    while (r && i < array.length) {
        x += array[i];
        if (r <= x) {
            return i;
        }
        i++;
    }

    return 0;
}

const Model = function Model () {};

Model.prototype.FMX = 0;
Model.prototype.FMY = 0;
Model.prototype.FMXxFMY = 0;
Model.prototype.T = 0;
Model.prototype.N = 0;

Model.prototype.initiliazedField = false;
Model.prototype.generationComplete = false;

Model.prototype.wave = null;
Model.prototype.compatible = null;
Model.prototype.weightLogWeights = null;
Model.prototype.sumOfWeights = 0;
Model.prototype.sumOfWeightLogWeights = 0;

Model.prototype.startingEntropy = 0;

Model.prototype.sumsOfOnes = null;
Model.prototype.sumsOfWeights = null;
Model.prototype.sumsOfWeightLogWeights = null;
Model.prototype.entropies = null;

Model.prototype.propagator = null;
Model.prototype.observed = null;
Model.prototype.distribution = null;

Model.prototype.stack = null;
Model.prototype.stackSize = 0;

Model.prototype.DX = [-1, 0, 1, 0];
Model.prototype.DY = [0, 1, 0, -1];
Model.prototype.opposite = [2, 3, 0, 1];

/**
 * @protected
 */
Model.prototype.initialize = function () {
    this.distribution = new Array(this.T);

    this.wave = new Array(this.FMXxFMY);
    this.compatible = new Array(this.FMXxFMY);

    for (let i = 0; i < this.FMXxFMY; i++) {
        this.wave[i] = new Array(this.T);
        this.compatible[i] = new Array(this.T);

        for (let t = 0; t < this.T; t++) {
            this.compatible[i][t] = [0,0,0,0];
        }
    }

    this.weightLogWeights = new Array(this.T);
    this.sumOfWeights = 0;
    this.sumOfWeightLogWeights = 0;

    for (let t = 0; t < this.T; t++) {
        this.weightLogWeights[t] = this.weights[t] * Math.log(this.weights[t]);
        this.sumOfWeights += this.weights[t];
        this.sumOfWeightLogWeights += this.weightLogWeights[t];
    }

    this.startingEntropy = Math.log(this.sumOfWeights) - this.sumOfWeightLogWeights / this.sumOfWeights;

    this.sumsOfOnes = new Array(this.FMXxFMY);
    this.sumsOfWeights = new Array(this.FMXxFMY);
    this.sumsOfWeightLogWeights = new Array(this.FMXxFMY);
    this.entropies = new Array(this.FMXxFMY);

    this.stack = new Array(this.FMXxFMY * this.T);
    this.stackSize = 0;
};

/**
 *
 * @param {Function} rng Random number generator function
 *
 * @returns {*}
 *
 * @protected
 */
Model.prototype.explore = function (x, y) {

    let min = 1000;
    let argmin = -1;
    const i = y * this.FMY + x;

    const amount = this.sumsOfOnes[i];

    if (amount === 0) return false;

    const entropy = this.entropies[i];

    if (amount > 1 && entropy <= min) {
        const noise = 0.000001 * Math.random();

        if (entropy + noise < min) {
            min = entropy + noise;
            argmin = i;
        }
    }

    if (argmin === -1) {
        this.observed = new Array(this.FMXxFMY);

        for (let t = 0; t < this.T; t++) {
            if (this.wave[i][t]) {
                this.observed[i] = t;
                break;
            }
        }

        return true;
    }

    for (let t = 0; t < this.T; t++) {
        this.distribution[t] = this.wave[argmin][t] ? this.weights[t] : 0;
    }
    const r = randomIndice(this.distribution, Math.random());

    const w = this.wave[argmin];
    for (let t = 0; t < this.T; t++) {
        if (w[t] !== (t === r)) this.ban(argmin, t);
    }

    return null;
};

/**
 * @protected
 */
Model.prototype.propagate = function () {
    while (this.stackSize > 0) {
        const e1 = this.stack[this.stackSize - 1];
        this.stackSize--;

        const i1 = e1[0];
        const x1 = i1 % this.FMX;
        const y1 = i1 / this.FMX | 0;

        for (let d = 0; d < 4; d++) {
            const dx = this.DX[d];
            const dy = this.DY[d];

            let x2 = x1 + dx;
            let y2 = y1 + dy;

            if (this.onBoundary(x2, y2)) continue;

            if (x2 < 0) x2 += this.FMX;
            else if (x2 >= this.FMX) x2 -= this.FMX;
            if (y2 < 0) y2 += this.FMY;
            else if (y2 >= this.FMY) y2 -= this.FMY;

            const i2 = x2 + y2 * this.FMX;
            const p = this.propagator[d][e1[1]];
            const compat = this.compatible[i2];

            for (let l = 0; l < p.length; l++) {
                const t2 = p[l];
                const comp = compat[t2];
                comp[d]--;
                if (comp[d] == 0) this.ban(i2, t2);
            }
        }
    }
};

/**
 * Execute a single iteration
 *
 * @param {Function} rng Random number generator function
 *
 * @returns {boolean|null}
 *
 * @protected
 */
Model.prototype.step = function (x, y) {
    this.explore(x, y);
    this.propagate();
    return null;
};

/**
 * Execute a fixed number of iterations. Stop when the generation is successful or reaches a contradiction.
 *
 * @param {int} [iterations=0] Maximum number of iterations to execute (0 = infinite)
 * @param {Function|null} [rng=Math.random] Random number generator function
 *
 * @returns {boolean} Success
 *
 * @public
 */
Model.prototype.walk = function (x, y) {
    if (!this.wave) this.initialize();

    if (!this.initiliazedField) {
        this.clear();
    }

    const result = this.step(x, y);

    if (result !== null) {
        return !!result;
    }

    return true;
};

/**
 * Check whether the previous generation completed successfully
 *
 * @returns {boolean}
 *
 * @public
 */
Model.prototype.isGenerationComplete = function () {
    return this.generationComplete;
};

/**
 *
 * @param {int} i
 * @param {int} t
 *
 * @protected
 */
Model.prototype.ban = function (i, t) {
    const comp = this.compatible[i][t];

    for (let d = 0; d < 4; d++) {
        comp[d] = 0;
    }

    this.wave[i][t] = false;

    this.stack[this.stackSize] = [i, t];
    this.stackSize++;

    this.sumsOfOnes[i] -= 1;
    this.sumsOfWeights[i] -= this.weights[t];
    this.sumsOfWeightLogWeights[i] -= this.weightLogWeights[t];

    const sum = this.sumsOfWeights[i];
    this.entropies[i] = Math.log(sum) - this.sumsOfWeightLogWeights[i] / sum;
};

/**
 * Clear the internal state to start a new generation
 *
 * @public
 */
Model.prototype.clear = function () {
    for (let i = 0; i < this.FMXxFMY; i++) {
        for (let t = 0; t < this.T; t++) {
            this.wave[i][t] = true;

            for (let d = 0; d < 4; d++) {
                this.compatible[i][t][d] = this.propagator[this.opposite[d]][t].length;
            }
        }

        this.sumsOfOnes[i] = this.weights.length;
        this.sumsOfWeights[i] = this.sumOfWeights;
        this.sumsOfWeightLogWeights[i] = this.sumOfWeightLogWeights;
        this.entropies[i] = this.startingEntropy;
    }

    this.initiliazedField = true;
    this.generationComplete = false;
};

/**
 *
 * @param {object} data Tiles, subset and constraints definitions
 * @param {string} subsetName Name of the subset to use from the data, use all tiles if falsy
 * @param {int} width The width of the generation
 * @param {int} height The height of the generation
 * @param {boolean} periodic Whether the source image is to be considered as periodic / as a repeatable texture
 *
 * @constructor
 */
const SimpleTiledModel = function SimpleTiledModel (data, subsetName, width, height, periodic) {
    const tilesize = data.tilesize || 16;

    this.FMX = width;
    this.FMY = height;
    this.FMXxFMY = width * height;

    this.periodic = periodic;
    this.tilesize = tilesize;

    const unique = !!data.unique;
    let subset = null;

    if (subsetName && data.subsets && !!data.subsets[subsetName]) {
        subset = data.subsets[subsetName];
    }

    const tile = function tile (f) {
        const result = new Array(tilesize * tilesize);

        for (let y = 0; y < tilesize; y++) {
            for (let x = 0; x < tilesize; x++) {
                result[x + y * tilesize] = f(x, y);
            }
        }

        return result;
    };

    const rotate = function rotate (array) {
        return tile(function (x, y) {
            return array[tilesize - 1 - y + x * tilesize];
        });
    };

    const reflect = function reflect(array) {
        return tile(function (x, y) {
            return array[tilesize - 1 - x + y * tilesize];
        });
    };

    this.tiles = [];
    const tempStationary = [];

    const action = [];
    const firstOccurrence = {};

    let funcA;
    let funcB;
    let cardinality;

    for (let i = 0; i < data.tiles.length; i++) {
        const currentTile = data.tiles[i];

        if (subset !== null && subset.indexOf(currentTile.name) === -1) {
            continue;
        }

        switch (currentTile.symmetry) {
            case 'L':
                cardinality = 4;
                funcA = function (i) {
                    return (i + 1) % 4;
                };
                funcB = function (i) {
                    return i % 2 === 0 ? i + 1 : i - 1;
                };
                break;
            case 'T':
                cardinality = 4;
                funcA = function (i) {
                    return (i + 1) % 4;
                };
                funcB = function (i) {
                    return i % 2 === 0 ? i : 4 - i;
                };
                break;
            case 'I':
                cardinality = 2;
                funcA = function (i) {
                    return 1 - i;
                };
                funcB = function (i) {
                    return i;
                };
                break;
            case '\\':
                cardinality = 2;
                funcA = function (i) {
                    return 1 - i;
                };
                funcB = function (i) {
                    return 1 - i;
                };
                break;
            case 'F':
                cardinality = 8;
                funcA = function (i) {
                    return i < 4 ? (i + 1) % 4 : 4 + (i - 1) % 4;
                };
                funcB = function (i) {
                    return i < 4 ? i + 4 : i - 4;
                };
                break;
            default:
                cardinality = 1;
                funcA = function (i) {
                    return i;
                };
                funcB = function (i) {
                    return i;
                };
                break;
        }

        this.T = action.length;
        firstOccurrence[currentTile.name] = this.T;

        for (let t = 0; t < cardinality; t++) {
            action.push([
                this.T + t,
                this.T + funcA(t),
                this.T + funcA(funcA(t)),
                this.T + funcA(funcA(funcA(t))),
                this.T + funcB(t),
                this.T + funcB(funcA(t)),
                this.T + funcB(funcA(funcA(t))),
                this.T + funcB(funcA(funcA(funcA(t))))
            ]);
        }


        let bitmap;

        if (unique) {
            for (let t = 0; t < cardinality; t++) {
                bitmap = currentTile.bitmap[t];
                this.tiles.push(tile(function (x, y) {
                    return [
                        bitmap[(tilesize * y + x) * 4],
                        bitmap[(tilesize * y + x) * 4 + 1],
                        bitmap[(tilesize * y + x) * 4 + 2],
                        bitmap[(tilesize * y + x) * 4 + 3]
                    ];
                }));
            }
        } else {
            bitmap = currentTile.bitmap;
            this.tiles.push(tile(function (x, y) {
                return [
                    bitmap[(tilesize * y + x) * 4],
                    bitmap[(tilesize * y + x) * 4 + 1],
                    bitmap[(tilesize * y + x) * 4 + 2],
                    bitmap[(tilesize * y + x) * 4 + 3]
                ];
            }));

            for (let t = 1; t < cardinality; t++) {
                this.tiles.push(t < 4 ? rotate(this.tiles[this.T + t - 1]) : reflect(this.tiles[this.T + t - 4]));
            }
        }

        for (let t = 0; t < cardinality; t++) {
            tempStationary.push(currentTile.weight || 1);
        }

    }

    this.T = action.length;
    this.weights = tempStationary;

    this.propagator = new Array(4);
    const tempPropagator = new Array(4);

    for (let i = 0; i < 4; i++) {
        this.propagator[i] = new Array(this.T);
        tempPropagator[i] = new Array(this.T);
        for (let t = 0; t < this.T; t++) {
            tempPropagator[i][t] = new Array(this.T);
            for (let t2 = 0; t2 < this.T; t2++) {
                tempPropagator[i][t][t2] = false;
            }
        }
    }

    for (let i = 0; i < data.neighbors.length; i++) {
        const neighbor = data.neighbors[i];

        const left = neighbor.left.split(' ').filter(function (v) {
            return v.length;
        });
        const right = neighbor.right.split(' ').filter(function (v) {
            return v.length;
        });

        if (subset !== null && (subset.indexOf(left[0]) === -1 || subset.indexOf(right[0]) === -1)) {
            continue;
        }

        const L = action[firstOccurrence[left[0]]][left.length == 1 ? 0 : parseInt(left[1], 10)];
        const D = action[L][1];
        const R = action[firstOccurrence[right[0]]][right.length == 1 ? 0 : parseInt(right[1], 10)];
        const U = action[R][1];

        tempPropagator[0][R][L] = true;
        tempPropagator[0][action[R][6]][action[L][6]] = true;
        tempPropagator[0][action[L][4]][action[R][4]] = true;
        tempPropagator[0][action[L][2]][action[R][2]] = true;

        tempPropagator[1][U][D] = true;
        tempPropagator[1][action[D][6]][action[U][6]] = true;
        tempPropagator[1][action[U][4]][action[D][4]] = true;
        tempPropagator[1][action[D][2]][action[U][2]] = true;
    }

    for (let t = 0; t < this.T; t++) {
        for (let t2 = 0; t2 < this.T; t2++) {
            tempPropagator[2][t][t2] = tempPropagator[0][t2][t];
            tempPropagator[3][t][t2] = tempPropagator[1][t2][t];
        }
    }

    for (let d = 0; d < 4; d++) {
        for (let t1 = 0; t1 < this.T; t1++) {
            const sp = [];
            const tp = tempPropagator[d][t1];

            for (let t2 = 0; t2 < this.T; t2++) {
                if (tp[t2]) {
                    sp.push(t2);
                }
            }

            this.propagator[d][t1] = sp;
        }
    }
};

SimpleTiledModel.prototype = Object.create(Model.prototype);
SimpleTiledModel.prototype.constructor = SimpleTiledModel;

/**
 *
 * @param {int} x
 * @param {int} y
 *
 * @returns {boolean}
 *
 * @protected
 */
SimpleTiledModel.prototype.onBoundary = function (x, y) {
    return !this.periodic && (x < 0 || y < 0 || x >= this.FMX || y >= this.FMY);
};

/**
 * Retrieve the RGBA data
 *
 * @param {Array|Uint8Array|Uint8ClampedArray} [array] Array to write the RGBA data into (must already be set to the correct size), if not set a new Uint8Array will be created and returned
 * @param {Array|Uint8Array|Uint8ClampedArray} [defaultColor] RGBA data of the default color to use on untouched tiles
 *
 * @returns {Array|Uint8Array|Uint8ClampedArray} RGBA data
 *
 * @public
 */
SimpleTiledModel.prototype.graphics = function (array, defaultColor) {
    array = array || new Uint8Array(this.FMXxFMY * this.tilesize * this.tilesize * 4);

    if (this.isGenerationComplete()) {
        this.graphicsComplete(array);
    } else {
        this.graphicsIncomplete(array, defaultColor);
    }

    return array;
};

/**
 * Set the RGBA data for a complete generation in a given array
 *
 * @param {Array|Uint8Array|Uint8ClampedArray} [array] Array to write the RGBA data into, if not set a new Uint8Array will be created and returned
 *
 * @protected
 */
SimpleTiledModel.prototype.graphicsComplete = function (array) {
    for (let x = 0; x < this.FMX; x++) {
        for (let y = 0; y < this.FMY; y++) {
            const tile = this.tiles[this.observed[x + y * this.FMX]];

            for (let yt = 0; yt < this.tilesize; yt++) {
                for (let xt = 0; xt < this.tilesize; xt++) {
                    const pixelIndex = (x * this.tilesize + xt + (y * this.tilesize + yt) * this.FMX * this.tilesize) * 4;
                    const color = tile[xt + yt * this.tilesize];

                    array[pixelIndex] = color[0];
                    array[pixelIndex + 1] = color[1];
                    array[pixelIndex + 2] = color[2];
                    array[pixelIndex + 3] = color[3];
                }
            }
        }
    }
};

/**
 * Set the RGBA data for an incomplete generation in a given array
 *
 * @param {Array|Uint8Array|Uint8ClampedArray} [array] Array to write the RGBA data into, if not set a new Uint8Array will be created and returned
 * @param {Array|Uint8Array|Uint8ClampedArray} [defaultColor] RGBA data of the default color to use on untouched tiles
 *
 * @protected
 */
SimpleTiledModel.prototype.graphicsIncomplete = function (array, defaultColor) {
    console.log(array);
    if (!defaultColor || defaultColor.length !== 4) {
        defaultColor = false;
    }

    for (let x = 0; x < this.FMX; x++) {
        for (let y = 0; y < this.FMY; y++) {
            const w = this.wave[x + y * this.FMX];
            let amount = 0;
            let sumWeights = 0;

            for (let t = 0; t < this.T; t++) {
                if (w[t]) {
                    amount++;
                    sumWeights += this.weights[t];
                }
            }

            const lambda = 1 / sumWeights;

            for (let yt = 0; yt < this.tilesize; yt++) {
                for (let xt = 0; xt < this.tilesize; xt++) {
                    const pixelIndex = (x * this.tilesize + xt + (y * this.tilesize + yt) * this.FMX * this.tilesize) * 4;

                    if (defaultColor && amount === this.T) {
                        array[pixelIndex] = defaultColor[0];
                        array[pixelIndex + 1] = defaultColor[1];
                        array[pixelIndex + 2] = defaultColor[2];
                        array[pixelIndex + 3] = defaultColor[3];
                    } else {
                        let r = 0;
                        let g = 0;
                        let b = 0;
                        let a = 0;

                        for (let t = 0; t < this.T; t++) {
                            if (w[t]) {
                                const c = this.tiles[t][xt + yt * this.tilesize];
                                const weight = this.weights[t] * lambda;
                                r+= c[0] * weight;
                                g+= c[1] * weight;
                                b+= c[2] * weight;
                                a+= c[3] * weight;
                            }
                        }

                        array[pixelIndex] = r;
                        array[pixelIndex + 1] = g;
                        array[pixelIndex + 2] = b;
                        array[pixelIndex + 3] = a;
                    }
                }
            }
        }
    }
};