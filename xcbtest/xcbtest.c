#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include <xcb/xcb.h>

int create_window(void)
{
  xcb_connection_t *connection = xcb_connect(NULL, NULL);

  const xcb_setup_t *setup = xcb_get_setup(connection);
  xcb_screen_iterator_t iter = xcb_setup_roots_iterator(setup);
  xcb_screen_t *screen = iter.data;

  xcb_window_t window = xcb_generate_id(connection);
  xcb_create_window(connection, XCB_COPY_FROM_PARENT, /* depth */
                    window,                           /* window id */
                    screen->root,                     /* parent window */
                    0, 0,                             /* x, y */
                    150, 150,                         /* width, height */
                    10,                               /* border width */
                    XCB_WINDOW_CLASS_INPUT_OUTPUT,    /* window class */
                    screen->root_visual,              /* visual */
                    0, NULL                           /* masks */
  );

  xcb_map_window(connection, window);

  xcb_flush(connection);

  pause();

  xcb_disconnect(connection);

  return 0;
}

int create_gcontext(void)
{
  xcb_connection_t *connection = xcb_connect(NULL, NULL);
  xcb_screen_t *screen =
      xcb_setup_roots_iterator(xcb_get_setup(connection)).data;

  xcb_drawable_t window = screen->root;
  xcb_gcontext_t gctx = xcb_generate_id(connection);
  uint32_t mask = XCB_GC_FOREGROUND;
  uint32_t value[] = {screen->black_pixel};

  xcb_create_gc(connection, gctx, window, mask, value);

  return 0;
}

int create_gcontext2(void)
{
  xcb_point_t points[] = {
      {10, 10},
      {10, 20},
      {20, 10},
      {20, 20},
  };

  xcb_point_t polyline[] = {
      {50, 10},
      {5, 20},
      {25, -20},
      {10, 10},
  };

  xcb_segment_t segments[] = {
      {100, 10, 140, 30},
      {110, 25, 130, 60},
  };

  xcb_rectangle_t rectangles[] = {
      {10, 50, 40, 20},
      {80, 50, 10, 40},
  };

  xcb_arc_t arcs[] = {
      {10, 100, 60, 40, 0, 90 << 6},
      {90, 100, 55, 40, 0, 270 << 6},
  };

  xcb_connection_t *connection = xcb_connect(NULL, NULL);

  xcb_screen_t *screen =
      xcb_setup_roots_iterator(xcb_get_setup(connection)).data;

  xcb_drawable_t window = screen->root;
  xcb_gcontext_t foreground = xcb_generate_id(connection);
  uint32_t mask = XCB_GC_FOREGROUND | XCB_GC_GRAPHICS_EXPOSURES;
  uint32_t values[2] = {screen->black_pixel, 0};

  xcb_create_gc(connection, foreground, window, mask, values);

  window = xcb_generate_id(connection);

  mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
  values[0] = screen->white_pixel;
  values[1] = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS;

  xcb_create_window(connection, XCB_COPY_FROM_PARENT, window, screen->root, 0,
                    0, 150, 150, 10, XCB_WINDOW_CLASS_INPUT_OUTPUT,
                    screen->root_visual, mask, values);

  // Enable close window event
  xcb_intern_atom_cookie_t cookie;
  xcb_intern_atom_reply_t* reply;

  cookie = xcb_intern_atom(connection, 1, 12, "WM_PROTOCOLS");
  reply = xcb_intern_atom_reply(connection, cookie, 0);

  cookie2 = xcb_intern_atom(connection, 0, 16, "WM_DELETE_WINDOW");
  reply2 = xcb_intern_atom_reply(connection, cookie2, 0);

  xcb_change_property(connection, XCB_PROP_MODE_REPLACE, window, reply->atom, 4, 32, 1, &reply2->atom);

  free(reply);
  free(reply2);

  // Set window type to splash/floating
  cookie = xcb_intern_atom(connection, 1, 19, "_NET_WM_WINDOW_TYPE");
  reply = xcb_intern_atom_reply(connection, cookie, 0);

  cookie2 = xcb_intern_atom(connection, 0, 26, "_NET_WM_WINDOW_TYPE_SPLASH");
  reply2 = xcb_intern_atom_reply(connection, cookie2, 0);

  xcb_change_property(connection, XCB_PROP_MODE_REPLACE, window, reply->atom, 4, 32, 1, &reply2->atom);

  free(reply);
  free(reply2);

  xcb_map_window(connection, window);
  xcb_flush(connection);

  xcb_generic_event_t *event;
  while ((event = xcb_wait_for_event(connection))) {
    switch (event->response_type & ~0x80) {
    case XCB_EXPOSE:
      xcb_poly_point(connection, XCB_COORD_MODE_ORIGIN, window, foreground, 4,
                     points);
      xcb_poly_line(connection, XCB_COORD_MODE_PREVIOUS, window, foreground, 4,
                    polyline);
      xcb_poly_segment(connection, window, foreground, 2, segments);
      xcb_poly_rectangle(connection, window, foreground, 2, rectangles);
      xcb_poly_arc(connection, window, foreground, 2, arcs);
      xcb_flush(connection);

      puts("expose");

      break;

    case XCB_CLIENT_MESSAGE:
      puts("client message");
      return 0;
      break;

    case XCB_KEY_PRESS:
      puts("key pressed");
      break;

    default:
      puts("default");
      break;
    }

    free(event);
  }

  return 0;
}

int main(void)
{
  return create_gcontext2();
}

