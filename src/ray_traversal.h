#pragma once


inline void traverseGrid(const glm::vec3& start, const glm::vec3& direction, const F& callback) {
    float tMin, tMax;

    if (!intersect_aabb_ray(start, direction, m_boundingBox, tMin, tMax)) {
        return;
    }

    glm::vec3 inversedirection = 1.f / direction;

    glm::vec3 deltaT;
    glm::vec3 nextCrossingT;

    int64_t cell[3];
    int64_t step[3];
    int64_t stop[3];

    for (size_t i = 0; i < 3; ++i) {
        float rayOrigCell = (start[i] + direction[i] * tMin) - m_leaves.origin()[i];
        cell[i] = std::clamp<int64_t>(static_cast<int64_t>(rayOrigCell / m_leaves.cellSize()[i]), 0, m_leaves.edgeDivider() - 1);

        if (direction[i] < 0) {
            deltaT[i] = -m_leaves.cellSize()[i] * inversedirection[i];
            nextCrossingT[i] = tMin + ((cell[i] + 0) * m_leaves.cellSize()[i] - rayOrigCell) * inversedirection[i];
            step[i] = -1;
            stop[i] = -1;
        }
        else {
            deltaT[i] = m_leaves.cellSize()[i] * inversedirection[i];
            nextCrossingT[i] = tMin + ((cell[i] + 1) * m_leaves.cellSize()[i] - rayOrigCell) * inversedirection[i];
            step[i] = 1;
            stop[i] = m_leaves.edgeDivider();
        }
    }

    uint8_t axis{};
    do {
        callback(cell[0], cell[1], cell[2]);

        const uint8_t k = ((nextCrossingT[0] < nextCrossingT[1]) << 2)
                        | ((nextCrossingT[0] < nextCrossingT[2]) << 1)
                        | ((nextCrossingT[1] < nextCrossingT[2]) << 0);

        const uint8_t arr[8]{ 2, 1, 2, 1, 2, 2, 0, 0 };
        axis = arr[k];

                 cell[axis] += step[axis];
        nextCrossingT[axis] += deltaT[axis];

    } while (cell[axis] != stop[axis]);
}
