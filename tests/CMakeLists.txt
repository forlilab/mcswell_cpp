#   Copyright (c) 2026 Scripps Research, Forli Lab.
#   All rights reserved.
#
#   This library is free software; you can redistribute it and/or
#   modify it under the terms of the GNU Lesser General Public
#   License as published by the Free Software Foundation; either
#   version 2.1 of the License, or (at your option) any later version.
#
#   This library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public
#   License along with this library; if not, write to the Free Software
#   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

add_executable(mcswell_analysis_tests
    analysis/test_main.cpp
    analysis/test_kdtree.cpp
    analysis/test_frame_extraction.cpp
)

target_link_libraries(mcswell_analysis_tests PRIVATE mcswell_analysis)

add_test(NAME mcswell_analysis_tests COMMAND mcswell_analysis_tests)
